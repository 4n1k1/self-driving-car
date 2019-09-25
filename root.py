import numpy

from matplotlib import rcParams
from matplotlib import pyplot
from matplotlib.lines import Line2D

from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.core.window import Window

from widgets import (
    Downtown,
    Airport,
    Painter,
    RGBAColor,
)
from car import Car


_PLOTTING_INTERVAL = 1000


class Root(Widget):
    def __init__(self):
        self.size = Window.size
        self.pos = (0, 0)

        self._cars = []
        self._airport = Airport()
        self._downtown = Downtown()
        self._painter = Painter()
        self._is_paused = True

        rcParams['toolbar'] = 'None'

        self._figure = pyplot.figure(figsize=(6, 3))
        self._figure.canvas.set_window_title("Mean Reward")

        self._plot = self._figure.add_subplot(1, 1, 1, facecolor=RGBAColor.GREY)
        self._empty_plot = self._figure.canvas.copy_from_bbox(self._plot.bbox)
        self._plot.grid()
        self._plot.set_xlabel("step")
        self._plot.set_ylabel("reward")
        self._plot.set_xlim(0.0, _PLOTTING_INTERVAL + 200)
        self._plot.set_ylim(-1.1, 1.1)
        self._plot_lines = []

        self._save_button = Button(
            text="save",
            pos=(Window.width - 270, 50),
            size=(100, 50),
        )

        self._load_button = Button(
            text="load",
            pos=(Window.width - 150, 50),
            size=(100, 50),
        )

        self._clear_button = Button(
            text="clear",
            pos=(Window.width - 150, 120),
            size=(100, 50),
        )

        self._pause_button = Button(
            text="run",
            pos=(Window.width - 270, 120),
            size=(100, 50),
        )

        Widget.__init__(self)

        self.sand = numpy.zeros(
            (self.width, self.height,)
        )

    @property
    def airport(self):
        return self._airport

    @property
    def downtown(self):
        return self._downtown

    def build(self, args):
        self.add_widget(self._airport)
        self.add_widget(self._downtown)
        self.add_widget(self._painter)

        for i in range(args.cars_count):
            car = Car(i, self._airport, args)
            car.build()

            self._cars.append(car)
            self.add_widget(car)

            plot_line = Line2D([], [], color=car.body_color)

            self._plot_lines.append(plot_line)
            self._plot.add_line(plot_line)

        self._figure.canvas.draw()

        pyplot.ion()
        pyplot.show()
        pyplot.pause(1.0/30)

        self.add_widget(self._save_button)
        self.add_widget(self._load_button)
        self.add_widget(self._clear_button)
        self.add_widget(self._pause_button)

        self._save_button.bind(on_release=self._save_brains)
        self._load_button.bind(on_release=self._load_brains)
        self._pause_button.bind(on_release=self._toggle_pause)
        self._clear_button.bind(on_release=self._clear_sand)

    def update(self, _):
        if self._is_paused:
            return

        for idx, car in enumerate(self._cars):
            car.move()

            plot_line = self._plot_lines[idx]
            plot_line.set_data(range(len(car.scores)), car.scores)

        self._figure.canvas.restore_region(self._empty_plot)

        for idx in range(len(self._cars)):
            plot_line = self._plot_lines[idx]
            self._plot.draw_artist(plot_line)

        self._figure.canvas.blit(self._plot.bbox)

    def _clear_sand(self, _):
        self._painter.canvas.clear()

        self.sand = numpy.zeros(
            (self.width, self.height,)
        )

    def _toggle_pause(self, _):
        self._is_paused = not self._is_paused

        if self._is_paused:
            self._pause_button.text = "run"
        else:
            self._pause_button.text = "pause"

    def _save_brains(self, _):
        for car in self._cars:
            car.save_brain()

    def _load_brains(self, _):
        for car in self._cars:
            car.load_brain()
