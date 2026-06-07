"""
DragClicker (Drag Click Limit)
==============================

Экспериментальная утилита: отслеживает вращение колёсика мыши, вычисляет на его
основе частоту кликов (CPS) и эмулирует реальные клики левой/правой кнопкой мыши
с этой частотой. Работает в фоне, глобально перехватывая события мыши через pynput.

Требования: Python 3.8+, pynput.
    pip install pynput

ВНИМАНИЕ: для работы подавления системной прокрутки (suppress=True) на Windows/Linux
могут потребоваться права администратора (root). В коде это не обрабатывается —
просто запустите программу с нужными правами при необходимости.

Запуск:
    python dclimit.py
"""

import os
import queue
import random
import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from tkinter import ttk

from pynput import mouse


# ---------------------------------------------------------------------------
# Константы и параметры по умолчанию
# ---------------------------------------------------------------------------

WHEEL_WINDOW_MS = 150          # скользящее окно измерения скорости колеса
UPDATE_INTERVAL_MS = 40        # период опроса очереди / перерисовки GUI
EMA_ALPHA = 0.3                # коэффициент экспоненциального сглаживания CPS
GRAPH_SECONDS = 10             # длина графика CPS (секунды)

# Точки линейной интерполяции "скорость колеса -> CPS"
WHEEL_LOW_TPS = 5.0            # 5 тиков/с
WHEEL_HIGH_TPS = 100.0        # 100 тиков/с

DEFAULT_MAX_CPS = 50.0         # CPS при WHEEL_HIGH_TPS (настраивается, 10..100)
DEFAULT_JITTER = 5.0           # амплитуда случайного джиттера ±CPS (0..15)
DEFAULT_INERTIA = 0.9          # коэффициент затухания CPS за такт (0.80..1.0)

# Цветовая схема (тёмная тема)
COL_BG = "#1e1f22"       # фон окна
COL_CARD = "#2b2d31"     # карточка-панель
COL_CARD2 = "#34373c"    # вложенный блок / трек ползунка
COL_FG = "#e8e8e8"       # основной текст
COL_MUTED = "#9aa0a6"    # приглушённый текст
COL_ACCENT = "#5aa9ff"   # акцент
COL_GOOD = "#3ecf8e"     # активно
COL_OFF = "#6b7280"      # отключено
COL_WARN = "#ff5d5d"     # предупреждение
COL_GRAPH = "#5aa9ff"    # линия графика
COL_GRID = "#3a3d42"     # сетка графика


def now_ms() -> float:
    """Текущее время в миллисекундах (монотонные часы)."""
    return time.monotonic() * 1000.0


def resource_path(name: str) -> str:
    """
    Абсолютный путь к ресурсу рядом с программой. Работает и при обычном
    запуске, и из собранного PyInstaller-онефайла (там ресурсы лежат во
    временной папке sys._MEIPASS).
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


# ---------------------------------------------------------------------------
# Модель данных
# ---------------------------------------------------------------------------

@dataclass
class DCLimitModel:
    """
    Хранит всё состояние программы. Доступ к простым float/bool полям считается
    атомарным (GIL), поэтому тяжёлых блокировок избегаем. Очередь событий колеса
    защищена собственным Lock, т.к. с ней работают несколько потоков.
    """

    # --- Состояние/режимы ---
    enabled: bool = False          # программа включена
    mode: str = "OFF"              # "LMB" / "RMB" / "OFF"

    # --- Флаги кнопок мыши ---
    x1_pressed: bool = False
    x2_pressed: bool = False
    middle_pressed: bool = False
    combo_ready: bool = True        # защита от повторного срабатывания X1+X2

    # --- Эмуляция ---
    emulating: bool = False         # сейчас выполняется собственный клик

    # --- CPS ---
    raw_cps: float = 0.0            # CPS до сглаживания
    smoothed_cps: float = 0.0       # CPS после EMA
    display_cps: float = 0.0        # итоговый CPS (для эмуляции и вывода)
    wheel_tps: float = 0.0          # текущая скорость колеса, тиков/с

    # --- Настройки (меняются ползунками на лету) ---
    max_cps: float = DEFAULT_MAX_CPS
    jitter: float = DEFAULT_JITTER
    inertia: float = DEFAULT_INERTIA

    # --- Статистика ---
    session_start: float = 0.0      # ms, начало текущей сессии
    cps_sum: float = 0.0            # для среднего
    cps_count: int = 0
    peak_cps: float = 0.0

    # --- Внутреннее ---
    _wheel_events: deque = field(default_factory=deque)  # отметки времени тиков (ms)
    _wheel_lock: threading.Lock = field(default_factory=threading.Lock)
    cps_history: deque = field(default_factory=lambda: deque(maxlen=512))  # (t_ms, cps)

    # ------------------------------------------------------------------ wheel
    def add_wheel_tick(self, t_ms: float) -> None:
        """Зарегистрировать один тик колеса."""
        with self._wheel_lock:
            self._wheel_events.append(t_ms)

    def _wheel_speed(self, t_ms: float) -> float:
        """Скорость колеса в тиках/с по скользящему окну WHEEL_WINDOW_MS."""
        with self._wheel_lock:
            # выкинуть устаревшие отметки
            while self._wheel_events and (t_ms - self._wheel_events[0]) > WHEEL_WINDOW_MS:
                self._wheel_events.popleft()
            ticks = len(self._wheel_events)
        return ticks / (WHEEL_WINDOW_MS / 1000.0)

    # -------------------------------------------------------------- интерполяция
    def _wheel_to_cps(self, tps: float) -> float:
        """
        Линейная интерполяция скорости колеса в целевой CPS:
            WHEEL_LOW_TPS  тик/с -> WHEEL_LOW_TPS  CPS (т.е. 5 -> 5)
            WHEEL_HIGH_TPS тик/с -> max_cps        CPS (т.е. 100 -> max_cps)
        Ниже нижней точки CPS = 0.
        """
        if tps < WHEEL_LOW_TPS:
            return 0.0
        if tps >= WHEEL_HIGH_TPS:
            return self.max_cps
        frac = (tps - WHEEL_LOW_TPS) / (WHEEL_HIGH_TPS - WHEEL_LOW_TPS)
        return WHEEL_LOW_TPS + frac * (self.max_cps - WHEEL_LOW_TPS)

    # ------------------------------------------------------------- основной апдейт
    def update_cps(self, t_ms: float) -> None:
        """Пересчитать CPS. Вызывается из главного потока каждые UPDATE_INTERVAL_MS."""
        # Пауза генерации, пока зажата X2 (режим обычного скролла).
        if not self.enabled or self.mode == "OFF" or self.x2_pressed:
            self.raw_cps = 0.0
            self.smoothed_cps = 0.0
            self.display_cps = 0.0
            self.wheel_tps = 0.0
            self._record_history(t_ms, 0.0)
            return

        tps = self._wheel_speed(t_ms)
        self.wheel_tps = tps
        target = self._wheel_to_cps(tps)

        if target > 0.0:
            # колесо крутится — берём целевой CPS
            self.raw_cps = target
        else:
            # инерция: колесо остановилось, CPS затухает
            self.raw_cps = self.smoothed_cps * self.inertia
            if self.raw_cps < 0.5:
                self.raw_cps = 0.0

        # EMA-сглаживание
        self.smoothed_cps = (EMA_ALPHA * self.raw_cps
                             + (1.0 - EMA_ALPHA) * self.smoothed_cps)

        # случайный джиттер ±jitter
        disp = self.smoothed_cps
        if disp > 0.0 and self.jitter > 0.0:
            disp += random.uniform(-self.jitter, self.jitter)
        self.display_cps = max(0.0, disp)

        self._update_stats(self.display_cps)
        self._record_history(t_ms, self.display_cps)

    # ---------------------------------------------------------------- статистика
    def _update_stats(self, cps: float) -> None:
        if cps > 0.0:
            self.cps_sum += cps
            self.cps_count += 1
            if cps > self.peak_cps:
                self.peak_cps = cps

    @property
    def avg_cps(self) -> float:
        return self.cps_sum / self.cps_count if self.cps_count else 0.0

    @property
    def session_seconds(self) -> float:
        if not self.enabled or self.session_start == 0.0:
            return 0.0
        return (now_ms() - self.session_start) / 1000.0

    def _record_history(self, t_ms: float, cps: float) -> None:
        self.cps_history.append((t_ms, cps))

    # ------------------------------------------------------------------- режимы
    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def toggle_mode(self, mode: str) -> None:
        """LMB/RMB взаимоисключающие; повторный выбор того же режима -> OFF."""
        self.mode = "OFF" if self.mode == mode else mode

    def reset_runtime(self) -> None:
        """Сброс при выключении: CPS, очередь событий, режим, статистика."""
        with self._wheel_lock:
            self._wheel_events.clear()
        self.raw_cps = 0.0
        self.smoothed_cps = 0.0
        self.display_cps = 0.0
        self.wheel_tps = 0.0
        self.mode = "OFF"
        self.cps_sum = 0.0
        self.cps_count = 0
        self.peak_cps = 0.0
        self.cps_history.clear()


# ---------------------------------------------------------------------------
# Поток-слушатель мыши
# ---------------------------------------------------------------------------

class MouseListener(threading.Thread):
    """
    Глобальный перехват событий мыши через pynput.mouse.Listener(suppress=True).
    Команды на изменение состояния передаются в главный поток через queue.Queue.
    """

    def __init__(self, model: DCLimitModel, cmd_queue: "queue.Queue"):
        super().__init__(daemon=True)
        self.model = model
        self.cmd_queue = cmd_queue
        self.listener: "mouse.Listener | None" = None

    # --------------------------------------------------------------- колбэки
    def on_click(self, x, y, button, pressed):
        m = self.model

        # Боковые кнопки -> отслеживаем флаги для комбо вкл/выкл
        if button == mouse.Button.x1:
            m.x1_pressed = pressed
            print(f"[mouse] X1 {'down' if pressed else 'up'}")
        elif button == mouse.Button.x2:
            m.x2_pressed = pressed
            print(f"[mouse] X2 {'down' if pressed else 'up'}")
        elif button == mouse.Button.middle:
            m.middle_pressed = pressed
            print(f"[mouse] MIDDLE {'down' if pressed else 'up'}")
        elif button == mouse.Button.left and pressed:
            # игнорируем собственные эмулированные клики
            if not m.emulating:
                self.cmd_queue.put(("toggle_mode", "LMB"))
        elif button == mouse.Button.right and pressed:
            if not m.emulating:
                self.cmd_queue.put(("toggle_mode", "RMB"))

        # Комбо X1+X2 -> вкл/выкл (однократно)
        if m.x1_pressed and m.x2_pressed and m.combo_ready:
            m.combo_ready = False
            print("[combo] X1+X2 -> toggle power")
            self.cmd_queue.put(("toggle_power", None))
        # разрешить повторное срабатывание только когда отпущена хотя бы одна
        if not (m.x1_pressed and m.x2_pressed):
            m.combo_ready = True

    def on_scroll(self, x, y, dx, dy):
        """
        Накопление тиков колеса для расчёта CPS (резервный путь / не-Windows).

        На Windows при включённой программе и отпущенной X2 событие колеса
        подавляется win32-фильтром ещё до вызова этого колбэка, поэтому тик там
        регистрируется прямо в фильтре (см. _win32_filter). Здесь тик
        учитывается на платформах без фильтра и как страховка, если событие
        всё-таки дошло (X2 зажата — обычный скролл, тик не нужен).
        """
        m = self.model
        if m.enabled and not m.x2_pressed and sys.platform != "win32":
            m.add_wheel_tick(now_ms())
        return None

    def _win32_filter(self, msg, data):
        """
        Низкоуровневый фильтр событий мыши (только Windows). Пока программа
        запущена и X2 НЕ зажата: регистрирует тик колеса (для расчёта CPS) и
        ПОЛНОСТЬЮ блокирует системную прокрутку (вертикальную и горизонтальную)
        через suppress_event(). При зажатой X2 событие не подавляется —
        работает обычный скролл.

        Важно: suppress_event() возбуждает исключение и прерывает дальнейшую
        обработку события в pynput, поэтому колбэк on_scroll при подавлении
        НЕ вызывается. Из-за этого тик нужно регистрировать ЗДЕСЬ, до подавления,
        иначе CPS всегда оставался бы нулевым и клики не генерировались.

        Для работы блокировки на Windows могут требоваться права администратора.
        """
        WM_MOUSEWHEEL = 0x020A
        WM_MOUSEHWHEEL = 0x020E
        if (self.model.enabled and not self.model.x2_pressed
                and msg in (WM_MOUSEWHEEL, WM_MOUSEHWHEEL)):
            self.model.add_wheel_tick(now_ms())
            if self.listener is not None:
                self.listener.suppress_event()

    # ----------------------------------------------------------------- запуск
    def run(self):
        kwargs = dict(on_click=self.on_click, on_scroll=self.on_scroll)
        # Точечное подавление прокрутки доступно через win32-фильтр (только Windows).
        # На прочих ОС прокрутка не блокируется (см. комментарий в шапке файла).
        if sys.platform == "win32":
            kwargs["win32_event_filter"] = self._win32_filter
        with mouse.Listener(**kwargs) as self.listener:
            self.listener.join()

    def stop(self):
        if self.listener is not None:
            self.listener.stop()


# ---------------------------------------------------------------------------
# Поток эмуляции кликов
# ---------------------------------------------------------------------------

class ClickEmulator(threading.Thread):
    """
    Бесконечный цикл: если программа включена, режим выбран и display_cps > 0 —
    выполняет реальный клик выбранной кнопкой и спит 1.0/cps секунд.
    Флаг emulating защищает от того, чтобы собственные клики меняли режим.
    """

    def __init__(self, model: DCLimitModel):
        super().__init__(daemon=True)
        self.model = model
        self.controller = mouse.Controller()
        self._running = True

    def run(self):
        m = self.model
        while self._running:
            cps = m.display_cps
            if m.enabled and m.mode in ("LMB", "RMB") and cps > 0.0 and not m.x2_pressed:
                button = mouse.Button.left if m.mode == "LMB" else mouse.Button.right
                m.emulating = True
                try:
                    self.controller.click(button)
                finally:
                    # небольшая пауза прежде чем снять флаг, чтобы событие успело пройти
                    m.emulating = False
                delay = 1.0 / max(1.0, cps)
                time.sleep(delay)
            else:
                time.sleep(0.01)

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Графический интерфейс
# ---------------------------------------------------------------------------

class DCLimitApp:
    """Главное окно Tkinter. Опрашивает очередь команд и перерисовывает GUI."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.model = DCLimitModel()
        self.cmd_queue: "queue.Queue" = queue.Queue()

        self.listener = MouseListener(self.model, self.cmd_queue)
        self.emulator = ClickEmulator(self.model)

        self._build_style()
        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.listener.start()
        self.emulator.start()

        self._tick()  # запуск цикла обновления

    # ----------------------------------------------------------------- стиль
    def _build_style(self):
        self.root.title("DragClicker")
        self.root.configure(bg=COL_BG)
        self._set_icon()

        # Размер окна подгоняем под экран и центрируем, чтобы приложение
        # всегда помещалось целиком.
        win_w, win_h = 470, 720
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        win_h = min(win_h, sh - 90)          # оставляем место под панель задач
        win_w = min(win_w, sw - 40)
        x = max(0, (sw - win_w) // 2)
        y = max(0, (sh - win_h) // 2 - 20)
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.minsize(440, 600)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=COL_BG)
        style.configure("Card.TFrame", background=COL_CARD)
        style.configure("TLabel", background=COL_BG, foreground=COL_FG)
        style.configure("Card.TLabel", background=COL_CARD, foreground=COL_FG)
        style.configure("Muted.TLabel", background=COL_CARD, foreground=COL_MUTED,
                        font=("Segoe UI", 9))
        style.configure("MutedBg.TLabel", background=COL_BG, foreground=COL_MUTED,
                        font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=COL_BG, foreground=COL_ACCENT,
                        font=("Segoe UI", 28, "bold"))
        style.configure("CPS.TLabel", background=COL_CARD, foreground=COL_ACCENT,
                        font=("Segoe UI", 54, "bold"))
        style.configure("Pill.TLabel", background=COL_CARD, foreground=COL_FG,
                        font=("Segoe UI", 12, "bold"))
        style.configure("Stat.TLabel", background=COL_CARD, foreground=COL_FG,
                        font=("Segoe UI", 11))

        # Кнопки
        style.configure("TButton", background=COL_CARD2, foreground=COL_FG,
                        borderwidth=0, focusthickness=0, padding=8,
                        font=("Segoe UI", 10, "bold"))
        style.map("TButton",
                  background=[("active", "#3e4147"), ("pressed", "#2a2c30")])
        style.configure("Accent.TButton", background=COL_ACCENT, foreground="#0c1116",
                        borderwidth=0, padding=8, font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton",
                  background=[("active", "#74b6ff"), ("pressed", "#4a93e6")])

        # Ползунки
        style.configure("Horizontal.TScale", background=COL_CARD,
                        troughcolor=COL_CARD2, borderwidth=0)

        # Тёмный скроллбар
        style.configure("Dark.Vertical.TScrollbar", background=COL_CARD2,
                        troughcolor=COL_BG, borderwidth=0, arrowsize=12,
                        arrowcolor=COL_MUTED)
        style.map("Dark.Vertical.TScrollbar",
                  background=[("active", COL_ACCENT)])

    def _set_icon(self):
        """Иконка окна из ico.ico рядом с программой (без падения, если её нет)."""
        path = resource_path("ico.ico")
        if not os.path.exists(path):
            print(f"[icon] ico.ico не найден: {path}")
            return
        try:
            self.root.iconbitmap(path)
        except tk.TclError as e:
            print(f"[icon] не удалось установить иконку: {e}")

    # -------------------------------------------------------------------- UI
    def _card(self):
        """Создаёт карточку-панель с внутренними отступами внутри прокручиваемой области."""
        outer = ttk.Frame(self.content, style="Card.TFrame")
        outer.pack(fill="x", padx=14, pady=6)
        inner = ttk.Frame(outer, style="Card.TFrame")
        inner.pack(fill="x", padx=14, pady=11)
        return inner

    def _build_scroll_area(self):
        """Прокручиваемая область — интерфейс не обрезается на малых экранах."""
        container = ttk.Frame(self.root, style="TFrame")
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, bg=COL_BG, highlightthickness=0)
        vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview,
                             style="Dark.Vertical.TScrollbar")
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.content = ttk.Frame(canvas, style="TFrame")
        win_id = canvas.create_window((0, 0), window=self.content, anchor="nw")
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win_id, width=e.width))
        self.content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        # Прокрутка колесом самой формы (удобно при настройке)
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

    def _draw_logo(self, c):
        """Векторный логотип: акцентный круг с молнией (быстрые авто-клики)."""
        c.create_oval(3, 4, 45, 46, fill="#1a2330", outline="")     # тень
        c.create_oval(2, 2, 44, 44, fill=COL_ACCENT, outline="")    # круг
        bolt = [26, 7, 16, 26, 23, 26, 20, 41, 34, 20, 26, 20]      # молния
        c.create_polygon(bolt, fill="#0c1116", outline="")

    def _build_ui(self):
        self._build_scroll_area()

        # --- Заголовок с логотипом ---
        header = ttk.Frame(self.content, style="TFrame")
        header.pack(fill="x", padx=16, pady=(16, 2))
        logo = tk.Canvas(header, width=48, height=48, bg=COL_BG,
                         highlightthickness=0)
        logo.pack(side="left", padx=(0, 12))
        self._draw_logo(logo)
        tbox = ttk.Frame(header, style="TFrame")
        tbox.pack(side="left", fill="x")
        ttk.Label(tbox, text="DragClicker", style="Title.TLabel").pack(anchor="w")
        ttk.Label(tbox, text="Drag Click Limit",
                  style="MutedBg.TLabel").pack(anchor="w")

        ttk.Label(self.content,
                  text="X1+X2 — вкл/выкл   •   ЛКМ/ПКМ — режим   •   удерживайте X2 для скролла",
                  style="MutedBg.TLabel").pack(anchor="w", padx=16, pady=(2, 4))

        # --- Блок состояния (две «таблетки») ---
        status = self._card()
        status.columnconfigure(0, weight=1)
        status.columnconfigure(1, weight=1)
        sbox = ttk.Frame(status, style="Card.TFrame")
        sbox.grid(row=0, column=0, sticky="w")
        ttk.Label(sbox, text="СТАТУС", style="Muted.TLabel").pack(anchor="w")
        self.lbl_status = ttk.Label(sbox, text="Отключено", style="Pill.TLabel",
                                    foreground=COL_OFF)
        self.lbl_status.pack(anchor="w")
        mbox = ttk.Frame(status, style="Card.TFrame")
        mbox.grid(row=0, column=1, sticky="e")
        ttk.Label(mbox, text="РЕЖИМ", style="Muted.TLabel").pack(anchor="e")
        self.lbl_mode = ttk.Label(mbox, text="OFF", style="Pill.TLabel",
                                  foreground=COL_OFF)
        self.lbl_mode.pack(anchor="e")

        # --- Крупный CPS ---
        cps_card = self._card()
        ttk.Label(cps_card, text="ТЕКУЩИЙ CPS", style="Muted.TLabel").pack()
        self.lbl_cps = ttk.Label(cps_card, text="0.0", style="CPS.TLabel")
        self.lbl_cps.pack()
        ttk.Label(cps_card, text="крутите колесо  •  X2 — обычный скролл",
                  style="Muted.TLabel").pack()

        # --- Статистика (сетка 2x2) ---
        stat = self._card()
        stat.columnconfigure(0, weight=1)
        stat.columnconfigure(1, weight=1)
        self.lbl_avg = self._stat_cell(stat, 0, 0, "Средний CPS", "0.0")
        self.lbl_peak = self._stat_cell(stat, 0, 1, "Пиковый CPS", "0.0")
        self.lbl_wheel = self._stat_cell(stat, 1, 0, "Колесо, тик/с", "0.0")
        self.lbl_time = self._stat_cell(stat, 1, 1, "Сессия, с", "0.0")

        # --- График CPS ---
        graph_card = self._card()
        ttk.Label(graph_card, text=f"CPS ЗА {GRAPH_SECONDS} С",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 6))
        self.canvas = tk.Canvas(graph_card, height=96, bg=COL_CARD2,
                                highlightthickness=0)
        self.canvas.pack(fill="x")

        # --- Кнопки управления ---
        btns = self._card()
        for c in range(2):
            btns.columnconfigure(c, weight=1)
        ttk.Button(btns, text="ВКЛ / ВЫКЛ", style="Accent.TButton",
                   command=lambda: self.cmd_queue.put(("toggle_power", None))
                   ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        ttk.Button(btns, text="Режим LMB",
                   command=lambda: self.cmd_queue.put(("toggle_mode", "LMB"))
                   ).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(btns, text="Режим RMB",
                   command=lambda: self.cmd_queue.put(("toggle_mode", "RMB"))
                   ).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(btns, text="Режим OFF",
                   command=lambda: self.cmd_queue.put(("set_mode", "OFF"))
                   ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=4)

        # --- Настройки (ползунки) ---
        settings = self._card()
        ttk.Label(settings, text="НАСТРОЙКИ", style="Muted.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        settings.columnconfigure(1, weight=1)

        self.var_max = tk.DoubleVar(value=self.model.max_cps)
        self.var_jit = tk.DoubleVar(value=self.model.jitter)
        self.var_iner = tk.DoubleVar(value=self.model.inertia)

        self._slider(settings, "Максимальный CPS", self.var_max, 10, 100, 1, self._on_max)
        self._slider(settings, "Джиттер ±CPS", self.var_jit, 0, 15, 2, self._on_jit)
        self._slider(settings, "Инерция", self.var_iner, 0.80, 1.0, 3, self._on_iner)

    def _stat_cell(self, parent, row, col, caption, value):
        cell = ttk.Frame(parent, style="Card.TFrame")
        cell.grid(row=row, column=col, sticky="w", padx=4, pady=5)
        ttk.Label(cell, text=caption, style="Muted.TLabel").pack(anchor="w")
        val = ttk.Label(cell, text=value, style="Stat.TLabel",
                        font=("Segoe UI", 14, "bold"))
        val.pack(anchor="w")
        return val

    def _slider(self, parent, label, var, frm, to, row, cb):
        ttk.Label(parent, text=label, style="Card.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=7)
        val = ttk.Label(parent, text=f"{var.get():.2f}", style="Card.TLabel",
                        foreground=COL_ACCENT, width=6, anchor="e")
        val.grid(row=row, column=2, sticky="e", padx=(8, 0))

        def on_change(_=None):
            val.configure(text=f"{var.get():.2f}")
            cb(var.get())

        ttk.Scale(parent, from_=frm, to=to, variable=var, orient="horizontal",
                  command=on_change).grid(row=row, column=1, sticky="ew", pady=7)

    # ----------------------------------------------------- колбэки настроек
    def _on_max(self, v):
        self.model.max_cps = float(v)

    def _on_jit(self, v):
        self.model.jitter = float(v)

    def _on_iner(self, v):
        self.model.inertia = float(v)

    # ----------------------------------------------- обработка команд из очереди
    def _process_commands(self):
        while True:
            try:
                cmd, arg = self.cmd_queue.get_nowait()
            except queue.Empty:
                break

            if cmd == "toggle_power":
                self._toggle_power()
            elif cmd == "toggle_mode":
                if self.model.enabled:
                    self.model.toggle_mode(arg)
            elif cmd == "set_mode":
                if self.model.enabled:
                    self.model.set_mode(arg)

    def _toggle_power(self):
        m = self.model
        if m.enabled:
            m.enabled = False
            m.reset_runtime()
            print("[power] OFF")
        else:
            m.enabled = True
            m.session_start = now_ms()
            m.cps_sum = 0.0
            m.cps_count = 0
            m.peak_cps = 0.0
            print("[power] ON")

    # ------------------------------------------------------------ главный цикл
    def _tick(self):
        self._process_commands()
        self.model.update_cps(now_ms())
        self._refresh_ui()
        self.root.after(UPDATE_INTERVAL_MS, self._tick)

    def _refresh_ui(self):
        m = self.model

        if m.enabled:
            self.lbl_status.configure(text="Активно", foreground=COL_GOOD)
        else:
            self.lbl_status.configure(text="Отключено", foreground=COL_OFF)

        mode_col = COL_ACCENT if m.mode in ("LMB", "RMB") else COL_OFF
        self.lbl_mode.configure(text=m.mode, foreground=mode_col)

        self.lbl_cps.configure(text=f"{m.display_cps:.1f}")
        self.lbl_avg.configure(text=f"{m.avg_cps:.1f}")
        self.lbl_peak.configure(text=f"{m.peak_cps:.1f}")
        self.lbl_wheel.configure(text=f"{m.wheel_tps:.1f}")
        self.lbl_time.configure(text=f"{m.session_seconds:.1f}")

        self._draw_graph()

    def _draw_graph(self):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width() or 420
        h = c.winfo_height() or 96
        t_now = now_ms()
        window = GRAPH_SECONDS * 1000.0

        # сетка
        for i in range(1, 4):
            y = h * i / 4
            c.create_line(0, y, w, y, fill=COL_GRID)

        # масштаб по Y — от max_cps + запас на джиттер
        y_max = max(10.0, self.model.max_cps + self.model.jitter)

        pts = [(t, cps) for (t, cps) in self.model.cps_history
               if t_now - t <= window]
        if len(pts) >= 2:
            coords = []
            for t, cps in pts:
                x = w * (1.0 - (t_now - t) / window)
                y = h - (min(cps, y_max) / y_max) * (h - 4) - 2
                coords.extend((x, y))
            c.create_line(*coords, fill=COL_GRAPH, width=2, smooth=True)

    # ------------------------------------------------------------- завершение
    def on_close(self):
        print("[app] shutting down...")
        self.model.enabled = False
        self.emulator.stop()
        self.listener.stop()
        self.root.after(50, self.root.destroy)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    DCLimitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
