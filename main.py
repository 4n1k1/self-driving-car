#!/usr/bin/python

import numpy
import argparse
import os

os.environ["KIVY_NO_ARGS"] = "1"

from kivy.app import App
from kivy.vector import Vector
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.graphics.context_instructions import PushMatrix, PopMatrix, Rotate, Translate
from kivy.clock import Clock
from kivy.core.window import Window

from ai import Brain
from matplotlib import pyplot

_DEFAULT_CARS_COUNT = 1
_CHECK_POINT_OFFEST = Vector(50, 50)
_PADDING = 25
_SIGNAL_RADIUS = 20
_SAND_LINE_RADIUS = 10

class _RGBAColor:
	RED = (1, 0, 0, 1)
	GREEN = (0, 1, 0, 1)
	BLUE = (0, 0, 1, 1)
	WHITE = (1, 1, 1, 1)
	GREY = (0.5, 0.5, 0.5, 1)

_IDX_TO_COLOR = {
	0: _RGBAColor.RED,
	1: _RGBAColor.GREEN,
	2: _RGBAColor.BLUE,
	3: _RGBAColor.WHITE,
	4: _RGBAColor.GREY,
}


class PositionMixin:
	@property
	def position(self):
		return Vector(*self.pos[:2])


class Car(RelativeLayout, PositionMixin):
	_ROTATIONS = (0, 20, -20)

	def __init__(self, car_idx, initial_destination):
		self.pos = (100, 100)

		self.idx = car_idx
		self.sand_speed = _PADDING / 4.0 - 5 + car_idx
		self.full_speed = _PADDING / 3.0 + car_idx
		self.velocity = self.full_speed
		self.action = 0
		self.direction = Vector(-1, 0)
		self.scores = []
		self.status_file = open("car{}_status".format(car_idx), "w")

		self.brain = Brain(5, 3)

		RelativeLayout.__init__(self)

		self.destination = initial_destination
		self.distance = self.position.distance(self.destination.position)

		self.body = Body(Vector(-5, -5), _IDX_TO_COLOR[self.idx])
		self.middle_sensor = Sensor(Vector(-30, -5), _RGBAColor.RED)
		self.right_sensor = Sensor(Vector(-20, 10), _RGBAColor.GREEN)
		self.left_sensor = Sensor(Vector(-20, -20), _RGBAColor.BLUE)
		self.center_mark = Center()

		self.add_widget(self.body)
		self.add_widget(self.middle_sensor)
		self.add_widget(self.right_sensor)
		self.add_widget(self.left_sensor)
		self.add_widget(self.center_mark)

		with self.canvas.before:
			PushMatrix()
			self.rotation = Rotate()

		with self.canvas.after:
			PopMatrix()

	def save_brain(self):
		self.brain.save("car{}_brain".format(self.idx))

	def load_brain(self):
		self.brain.load("car{}_brain".format(self.idx))

	def _rotate(self, angle_of_rotation):
		self.rotation.angle += angle_of_rotation

		self.direction = self.direction.rotate(angle_of_rotation)

	def move(self):
		self.status_file.seek(0)

		self.status_file.write(
			"Current destination [{}, [{:>4d}, {:>4d}]]\n".format(
				self.destination,
				self.destination.position.x,
				self.destination.position.y,
			)
		)

		angle_of_rotation = self._ROTATIONS[self.action]

		self._rotate(angle_of_rotation)

		self.middle_sensor.rotate(angle_of_rotation)
		self.left_sensor.rotate(angle_of_rotation)
		self.right_sensor.rotate(angle_of_rotation)

		self.pos = self.direction * self.velocity + self.pos

		distance = self.position.distance(self.destination.position)

		self.status_file.write("Distance    {:>9.4f}\n".format(distance))

		self.action = self.brain.update(
			self._get_reward(distance),
			self._get_state(),
		)
		self.distance = distance

		if self.distance < _PADDING * 2:
			if isinstance(self.destination, Airport):
				self.destination = self.parent.downtown
			else:
				self.destination = self.parent.airport

		self.scores.append(self.brain.score())

		if len(self.scores) > 1000:
			del self.scores[0]

	def _get_state(self):
		self.orientation = self.direction.angle(self.destination.position - self.position)/180.

		self.status_file.write("Orientation {:>9.4f}\n".format(self.orientation))

		self.left_sensor.signal = int(
			numpy.sum(
				self.parent.sand[
					int(self.left_sensor.abs_pos.x) - _SIGNAL_RADIUS : int(self.left_sensor.abs_pos.x) + _SIGNAL_RADIUS,
					int(self.left_sensor.abs_pos.y) - _SIGNAL_RADIUS : int(self.left_sensor.abs_pos.y) + _SIGNAL_RADIUS,
				]
			)
		) / 400.
		self.middle_sensor.signal = int(
			numpy.sum(
				self.parent.sand[
					int(self.middle_sensor.abs_pos.x) - _SIGNAL_RADIUS : int(self.middle_sensor.abs_pos.x) + _SIGNAL_RADIUS,
					int(self.middle_sensor.abs_pos.y) - _SIGNAL_RADIUS : int(self.middle_sensor.abs_pos.y) + _SIGNAL_RADIUS,
				]
			)
		) / 400.
		self.right_sensor.signal = int(
			numpy.sum(
				self.parent.sand[
					int(self.right_sensor.abs_pos.x) - _SIGNAL_RADIUS : int(self.right_sensor.abs_pos.x) + _SIGNAL_RADIUS,
					int(self.right_sensor.abs_pos.y) - _SIGNAL_RADIUS : int(self.right_sensor.abs_pos.y) + _SIGNAL_RADIUS,
				]
			)
		) / 400.

		self._set_collision_signal_value(self.left_sensor)
		self._set_collision_signal_value(self.right_sensor)
		self._set_collision_signal_value(self.middle_sensor)

		self.status_file.write(
			"Right sensor  [{: 2.4f}, [{:>9.4f}, {:>9.4f}]]\n".format(
				self.right_sensor.signal,
				self.right_sensor.abs_pos.x,
				self.right_sensor.abs_pos.y,
			)
		)
		self.status_file.write(
			"Left sensor   [{: 2.4f}, [{:>9.4f}, {:>9.4f}]]\n".format(
				self.left_sensor.signal,
				self.left_sensor.abs_pos.x,
				self.left_sensor.abs_pos.y,
			)
		)
		self.status_file.write(
			"Middle sensor [{: 2.4f}, [{:>9.4f}, {:>9.4f}]]\n".format(
				self.middle_sensor.signal,
				self.middle_sensor.abs_pos.x,
				self.middle_sensor.abs_pos.y,
			)
		)

		return (
			self.left_sensor.signal,
			self.middle_sensor.signal,
			self.right_sensor.signal,
			self.orientation,
			-self.orientation,
		)

	def _set_collision_signal_value(self, sensor):
		if (
			sensor.abs_pos.x >= self.parent.width - _PADDING or
			sensor.abs_pos.x <= _PADDING or
			sensor.abs_pos.y >= self.parent.height - _PADDING or
			sensor.abs_pos.y <= _PADDING
		):
			sensor.signal = 1.

	def _get_reward(self, distance):
		reward = 0.0

		if self.parent.sand[int(self.position.x), int(self.position.y)] > 0:
			self.velocity = self.sand_speed
			reward = -0.5
		else:
			self.velocity = self.full_speed

			if distance < self.distance:
				reward = 0.1
			else:
				if 0.0 < self.right_sensor.signal < 1.0:
					reward = 0.1
				else:
					reward = -0.2

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

		self.status_file.write("Reward      {: >9.4f}\n".format(reward))

		return reward


class Center(Widget):
	def __init__(self):
		Widget.__init__(self)

		self.pos = (0, 0)
		self.size = (1, 1)

		with self.canvas:
			Color(*_RGBAColor.GREY)
			Ellipse(pos=self.pos, size=self.size)


class Downtown(Widget, PositionMixin):
	def __init__(self):
		self.pos = Vector(*_CHECK_POINT_OFFEST)
		self.size = (20, 20)

		Widget.__init__(self)

		with self.canvas:
			Color(*_RGBAColor.GREY)
			Rectangle(pos=self.pos, size=self.size)

	def __repr__(self):
		return "Downtown"


class Airport(Widget, PositionMixin):
	def __init__(self):
		self.pos = Vector(*Window.size) - Vector(*_CHECK_POINT_OFFEST)
		self.size = (20, 20)

		Widget.__init__(self)

		with self.canvas:
			Color(*_RGBAColor.GREY)
			Rectangle(pos=self.pos, size=self.size)

	def __repr__(self):
		return " Airport"

class Body(Widget, PositionMixin):
	def __init__(self, pos, color):
		self.color = color
		self.pos = pos
		self.size = (20, 10)

		Widget.__init__(self)

		with self.canvas:
			Color(*self.color)
			Rectangle(pos=self.pos, size=self.size)


class Sensor(Widget):
	def __init__(self, pos, color):
		self.color = color
		self.pos = pos
		self.size = (10, 10)
		self.signal = 0.

		self._position = Vector(*self.pos)

		Widget.__init__(self)

		with self.canvas:
			Color(*self.color)
			Ellipse(pos=self.pos, size=self.size)

	def rotate(self, angle_of_rotation):
		self._position = self._position.rotate(angle_of_rotation)

	@property
	def abs_pos(self):
		return self.parent.position + self._position


class Painter(Widget):

	def __init__(self):
		Widget.__init__(self)

		self.__last_x = 0
		self.__last_y = 0
		self.__length = 0
		self.__n_points = 0

	def on_touch_down(self, touch):
		with self.canvas:
			Color(0.8,0.7,0)
			touch.ud['line'] = Line(points = (touch.x, touch.y), width = _PADDING)

			self.__last_x = int(touch.x)
			self.__last_y = int(touch.y)
			self.__n_points = 0
			self.__length = 0

			self.parent.sand[
				int(touch.x) - _SAND_LINE_RADIUS : int(touch.x) + _SAND_LINE_RADIUS,
				int(touch.y) - _SAND_LINE_RADIUS : int(touch.y) + _SAND_LINE_RADIUS
			] = 1

	def on_touch_move(self, touch):
		if touch.button == 'left':
			touch.ud['line'].points += [touch.x, touch.y]
			touch.ud['line'].width = _SAND_LINE_RADIUS * 2
			self.parent.sand[
				int(touch.x) - _SAND_LINE_RADIUS : int(touch.x) + _SAND_LINE_RADIUS,
				int(touch.y) - _SAND_LINE_RADIUS : int(touch.y) + _SAND_LINE_RADIUS
			] = 1
			self.__last_x = touch.x
			self.__last_y = touch.y


class Map(Widget):

	def __init__(self):
		self.size = Window.size
		self.pos = (0, 0)

		self.__cars = []
		self.__airport = Airport()
		self.__downtown = Downtown()

		self.__painter = Painter()
		self.__is_paused = True

		self.__plot_button = Button(
			text='plot',
			pos=(Window.width - 150, 50),
			size=(100, 50),
		)

		self.__save_button = Button(
			text="save",
			pos=(Window.width - 270, 50),
			size=(100, 50),
		)

		self.__load_button = Button(
			text="load",
			pos=(Window.width - 390, 50),
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

		self.__save_map_button = Button(
			text="screen",
			pos=(Window.width - 390, 120),
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
		self.add_widget(self.__plot_button)
		self.add_widget(self.__save_button)
		self.add_widget(self.__load_button)
		self.add_widget(self.__clear_button)
		self.add_widget(self.__pause_button)
		self.add_widget(self.__save_map_button)

		for i in range(0, cars_count):
			car = Car(i, self.__airport)

			self.__cars.append(car)
			self.add_widget(car)

		self.__plot_button.bind(on_release=self.__plot_learning_process)
		self.__save_button.bind(on_release=self.__save_brains)
		self.__load_button.bind(on_release=self.__load_brains)
		self.__pause_button.bind(on_release=self.__toggle_pause)
		self.__clear_button.bind(on_release=self.__clear_sand)
		self.__save_map_button.bind(on_release=self.__save_map)

	def update(self, dt):
		if self.__is_paused:
			return

		for car in self.__cars:
			car.move()

	def __save_map(self, save_map_button):
		self.export_to_png("sdc.png")

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

	def __plot_learning_process(self, plot_button):
		for idx, car in enumerate(self.__cars):
			pyplot.plot(car.scores, color=car.body.color)

		pyplot.show()

	def __save_brains(self, save_button):
		for car in self.__cars:
			car.save_brain()

	def __load_brains(self, load_button):
		for car in self.__cars:
			car.load_brain()



class CarApp(App):
	def __init__(self):
		App.__init__(self)

		self.__map = Map()

	def build(self):
		parser = argparse.ArgumentParser()
		parser.add_argument(
			"-c", "--cars_count",
			help="display a square of a given number",
			type=int,
			default=_DEFAULT_CARS_COUNT
		)
		args = parser.parse_args()

		self.__map.build(args.cars_count)

		Clock.schedule_interval(self.__map.update, 1.0/60.0)

		return self.__map


if __name__ == '__main__':
    CarApp().run()
