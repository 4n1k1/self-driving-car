import numpy
import matplotlib
import warnings

from matplotlib import rcParams
from matplotlib import pyplot
from matplotlib.lines import Line2D

from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.relativelayout import RelativeLayout
from kivy.graphics.context_instructions import PushMatrix, PopMatrix, Rotate
from kivy.core.window import Window

from ai import Brain
from widgets import (
	Center,
	Body,
	Sensor,
	Downtown,
	Airport,
	Painter,
	RGBAColor,
	PositionMixin,
)

_PADDING = 25
_SIGNAL_RADIUS = 25
_PLOTTING_INTERVAL = 1000
_IDX_TO_COLOR = {
	0: (RGBAColor.RED,    "red   "),
	1: (RGBAColor.GREEN,  "green "),
	2: (RGBAColor.BLUE,   "blue  "),
	3: (RGBAColor.PURPLE, "purple"),
	4: (RGBAColor.YELLOW, "yellow"),
	5: (RGBAColor.CYAN,   "cyan"),
}


class Car(RelativeLayout):
	_ROTATIONS = (0, 20, -20)

	def __init__(self, car_idx, initial_destination):
		self.pos = (100, 100)

		self._idx = car_idx
		self._sand_speed = _PADDING / 5.0
		self._full_speed = _PADDING / 4.0
		self._velocity = self._full_speed
		self._last_action = 0  # index of _ROTATIONS
		self._direction = Vector(-1, 0)
		self._scores = []
		self._orientation = 0.0
		self._distance = 0.0
		self._status_file = open("car{}_status".format(car_idx), "w")
		self._current_destination = initial_destination

		RelativeLayout.__init__(self)

		with self.canvas.before:
			PushMatrix()
			self._rotation = Rotate()

		with self.canvas.after:
			PopMatrix()

		self._center = Center()
		self._body = Body(Vector(-5, -5), _IDX_TO_COLOR[self._idx][0])
		self._middle_sensor = Sensor(Vector(-30, -5), RGBAColor.RED, self._rotation)
		self._right_sensor = Sensor(Vector(-20, 10), RGBAColor.GREEN, self._rotation)
		self._left_sensor = Sensor(Vector(-20, -20), RGBAColor.BLUE, self._rotation)
		self._brain = Brain(len(self._state), len(self._ROTATIONS))

	def build(self):
		self.add_widget(self._body)
		self.add_widget(self._middle_sensor)
		self.add_widget(self._right_sensor)
		self.add_widget(self._left_sensor)
		self.add_widget(self._center)

	@property
	def _state(self):
		return (
			self._left_sensor.signal,
			self._middle_sensor.signal,
			self._right_sensor.signal,
			self._orientation,
			-self._orientation,
		)

	@property
	def position(self):
		return Vector(*self.pos) + self._center.position

	def _rotate(self, angle_of_rotation):
		self._rotation.angle += angle_of_rotation

		self._direction = self._direction.rotate(angle_of_rotation)

	def _write_status(self, reward):
		self._status_file.seek(0)
		self._status_file.write(
			"Car color    : {}\n".format(_IDX_TO_COLOR[self._idx][1])
		)
		self._status_file.write(
			"Destination  : {}, ({:>4d}, {:>4d})\n".format(
				self._current_destination,
				self._current_destination.position.x,
				self._current_destination.position.y,
			)
		)
		self._status_file.write(
			"Distance     : {:>9.4f}\n".format(self._distance)
		)
		self._status_file.write(
			"Orientation  : {:>9.4f}\n".format(self._orientation)
		)
		self._status_file.write(
			"Reward       : {: >9.4f}\n".format(reward)
		)
		self._status_file.write(
			"Middle sensor: {: 2.4f}, ({:>9.4f}, {:>9.4f})\n".format(
				self._middle_sensor.signal,
				self._middle_sensor.abs_pos.x,
				self._middle_sensor.abs_pos.y,
			)
		)
		self._status_file.write(
			"Right sensor : {: 2.4f}, ({:>9.4f}, {:>9.4f})\n".format(
				self._right_sensor.signal,
				self._right_sensor.abs_pos.x,
				self._right_sensor.abs_pos.y,
			)
		)
		self._status_file.write(
			"Left sensor  : {: 2.4f}, ({:>9.4f}, {:>9.4f})\n".format(
				self._left_sensor.signal,
				self._left_sensor.abs_pos.x,
				self._left_sensor.abs_pos.y,
			)
		)

	def _set_collision_signal_value(self, sensor):
		if (
			sensor.abs_pos.x >= self.parent.width - _PADDING or
			sensor.abs_pos.x <= _PADDING or
			sensor.abs_pos.y >= self.parent.height - _PADDING or
			sensor.abs_pos.y <= _PADDING
		):
			sensor.signal = 1.

	def _get_reward(self, approached_destination):
		reward = 0.0

		if self.parent.sand[int(self.position.x), int(self.position.y)] > 0:
			self._velocity = self._sand_speed
			reward = -0.8
		else:
			self._velocity = self._full_speed

			if (
				approached_destination or
				(
					self._right_sensor.signal > 0.0 and
					self._left_sensor.signal == 0.0
				)
			):
				reward = 0.2
			else:
				reward = -0.5

		if self.position.x < _PADDING:
			self.pos = (_PADDING, self.pos[1])
			reward = -1.0

		if self.position.x > self.parent.width - _PADDING:
			self.pos = (self.parent.width - _PADDING, self.pos[1])
			reward = -1.0

		if self.position.y < _PADDING:
			self.pos = (self.pos[0], _PADDING)
			reward = -1.0

		if self.position.y > self.parent.height - _PADDING:
			self.pos = (self.pos[0], self.parent.height - _PADDING)
			reward = -1.0

		return reward

	def move(self):
		self._rotate(self._ROTATIONS[self._last_action])

		self.pos = self._direction * self._velocity + self.pos

		new_distance = self.position.distance(self._current_destination.position)

		self._orientation = self._direction.angle(self._current_destination.position - self.position)/180.
		self._left_sensor.signal = numpy.sum(
			self.parent.sand[
				int(self._left_sensor.abs_pos.x) - _SIGNAL_RADIUS : int(self._left_sensor.abs_pos.x) + _SIGNAL_RADIUS,
				int(self._left_sensor.abs_pos.y) - _SIGNAL_RADIUS : int(self._left_sensor.abs_pos.y) + _SIGNAL_RADIUS,
			]
		) / 400.
		self._middle_sensor.signal = numpy.sum(
			self.parent.sand[
				int(self._middle_sensor.abs_pos.x) - _SIGNAL_RADIUS : int(self._middle_sensor.abs_pos.x) + _SIGNAL_RADIUS,
				int(self._middle_sensor.abs_pos.y) - _SIGNAL_RADIUS : int(self._middle_sensor.abs_pos.y) + _SIGNAL_RADIUS,
			]
		) / 400.
		self._right_sensor.signal = numpy.sum(
			self.parent.sand[
				int(self._right_sensor.abs_pos.x) - _SIGNAL_RADIUS : int(self._right_sensor.abs_pos.x) + _SIGNAL_RADIUS,
				int(self._right_sensor.abs_pos.y) - _SIGNAL_RADIUS : int(self._right_sensor.abs_pos.y) + _SIGNAL_RADIUS,
			]
		) / 400.

		self._set_collision_signal_value(self._left_sensor)
		self._set_collision_signal_value(self._right_sensor)
		self._set_collision_signal_value(self._middle_sensor)

		reward = self._get_reward(new_distance <= self._distance)

		self._last_action = self._brain.update(
			reward,
			self._state,
		)

		self._distance = new_distance

		if self._distance < _PADDING * 2:
			if isinstance(self._current_destination, Airport):
				self._current_destination = self.parent.downtown
			else:
				self._current_destination = self.parent.airport

		self._scores.append(self._brain.score)

		if len(self._scores) > 1000:
			del self._scores[0]

		self._write_status(reward)

	def save_brain(self):
		self._brain.save("car{}_brain".format(self._idx))

	def load_brain(self):
		self._brain.load("car{}_brain".format(self._idx))

	@property
	def scores(self):
		return self._scores

	@property
	def body_color(self):
		return self._body.color


class Root(Widget):
	def __init__(self):
		self.size = Window.size
		self.pos = (0, 0)

		self.__cars = []
		self.__airport = Airport()
		self.__downtown = Downtown()
		self.__painter = Painter()
		self.__is_paused = True

		with warnings.catch_warnings():
			rcParams['toolbar'] = 'None'
			warnings.simplefilter("ignore")
			self.__figure = pyplot.figure(figsize=(6, 3))

		self.__plot = self.__figure.add_subplot(1, 1, 1, axisbg=RGBAColor.GREY)
		self.__plot.grid()
		self.__plot.set_xlabel("step")
		self.__plot.set_ylabel("reward")
		self.__plot.set_xlim( 0.0, _PLOTTING_INTERVAL + 200)
		self.__plot.set_ylim(-1.3, 0.7)
		self.__plot_lines = []

		self.__save_button = Button(
			text="save",
			pos=(Window.width - 270, 50),
			size=(100, 50),
		)

		self.__load_button = Button(
			text="load",
			pos=(Window.width - 150, 50),
			size=(100, 50),
		)

		self.__clear_button = Button(
			text="clear",
			pos=(Window.width - 150, 120),
			size=(100, 50),
		)

		self.__pause_button = Button(
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
		return self.__airport

	@property
	def downtown(self):
		return self.__downtown

	def build(self, cars_count):
		self.add_widget(self.__airport)
		self.add_widget(self.__downtown)
		self.add_widget(self.__painter)

		for i in range(0, cars_count):
			car = Car(i, self.__airport)
			car.build()

			self.__cars.append(car)
			self.add_widget(car)

			plot_line = Line2D([], [], color=car.body_color)

			self.__plot_lines.append(plot_line)
			self.__plot.add_line(plot_line)

		pyplot.show(block=False)

		self.__empty_plot = self.__figure.canvas.copy_from_bbox(self.__plot.bbox)

		self.add_widget(self.__save_button)
		self.add_widget(self.__load_button)
		self.add_widget(self.__clear_button)
		self.add_widget(self.__pause_button)

		self.__save_button.bind(on_release=self.__save_brains)
		self.__load_button.bind(on_release=self.__load_brains)
		self.__pause_button.bind(on_release=self.__toggle_pause)
		self.__clear_button.bind(on_release=self.__clear_sand)

	def update(self, dt):
		if self.__is_paused:
			return

		for idx, car in enumerate(self.__cars):
			car.move()

			plot_line = self.__plot_lines[idx]
			plot_line.set_data(range(len(car.scores)), car.scores)

		self.__figure.canvas.restore_region(self.__empty_plot)

		for idx in range(len(self.__cars)):
			plot_line = self.__plot_lines[idx]
			self.__plot.draw_artist(plot_line)

		self.__figure.canvas.blit(self.__plot.bbox)

	def __clear_sand(self, clear_button):
		self.__painter.canvas.clear()

		self.sand = numpy.zeros(
			(self.width, self.height,)
		)

	def __toggle_pause(self, pause_button):
		self.__is_paused = not self.__is_paused

		if self.__is_paused:
			self.__pause_button.text = "run"
		else:
			self.__pause_button.text = "pause"

	def __save_brains(self, save_button):
		for car in self.__cars:
			car.save_brain()

	def __load_brains(self, load_button):
		for car in self.__cars:
			car.load_brain()
