#!/usr/bin/python

import numpy

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

_CARS_COUNT = 3
_CHECK_POINT_OFFEST = Vector(50, 50)
_PADDING = 20

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

	def __init__(self, car_idx, map_widget):
		self.idx = car_idx
		self.pos = Vector((car_idx + 1) * 250, (car_idx + 1) * 250)
		self.sand_speed = _PADDING / 1.5 - 5 + car_idx
		self.full_speed = _PADDING / 1.5 + car_idx
		self.velocity = self.full_speed
		self.action = 0
		self.direction = Vector(-1, 0)
		self.reward = 0.0
		self.state = (0, 0, 0, 0, 0)
		self.scores = []

		self.brain = Brain(
			len(self.state),
			len(self._ROTATIONS),
		)

		RelativeLayout.__init__(self)

		self.destination = map_widget.airport
		self.distance = self.position.distance(self.destination.position)

		self.body = Body(Vector(-5, -5), _IDX_TO_COLOR[self.idx])
		self.middle_sensor = Sensor(Vector(-20, -5), _RGBAColor.RED)
		self.right_sensor = Sensor(Vector(-12, 10), _RGBAColor.GREEN)
		self.left_sensor = Sensor(Vector(-12, -20), _RGBAColor.BLUE)
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

	def move(self):
		angle_of_rotation = self._ROTATIONS[self.action]
		self.rotation.angle += angle_of_rotation
		self.direction = self.direction.rotate(angle_of_rotation)

		self.pos = self.direction * self.velocity + self.pos

		distance = self.position.distance(self.destination.position)

		self.action = self.brain.update(
			self._get_reward(distance),
			self._get_state(),
		)
		self.distance = distance

		if self.distance < _PADDING:
			if isinstance(self.destination, Airport):
				self.destination = self.parent.downtown
			else:
				self.destination = self.parent.airport

		self.scores.append(self.brain.score())

		if len(self.scores) > 1000:
			del self.scores[0]

	def _get_state(self):
		self.orientation = self.direction.angle(self.destination.position - self.position)/180.
		self.left_sensor.signal = int(
			numpy.sum(
				self.parent.sand[
					int(self.left_sensor.abs_pos.x) - _PADDING : int(self.left_sensor.abs_pos.x) + _PADDING,
					int(self.left_sensor.abs_pos.y) - _PADDING : int(self.left_sensor.abs_pos.y) + _PADDING,
				]
			)
		) / 400.
		self.middle_sensor.signal = int(
			numpy.sum(
				self.parent.sand[
					int(self.middle_sensor.abs_pos.x) - _PADDING : int(self.middle_sensor.abs_pos.x) + _PADDING,
					int(self.middle_sensor.abs_pos.y) - _PADDING : int(self.middle_sensor.abs_pos.y) + _PADDING,
				]
			)
		) / 400.
		self.right_sensor.signal = int(
			numpy.sum(
				self.parent.sand[
					int(self.right_sensor.abs_pos.x) - _PADDING : int(self.right_sensor.abs_pos.x) + _PADDING,
					int(self.right_sensor.abs_pos.y) - _PADDING : int(self.right_sensor.abs_pos.y) + _PADDING,
				]
			)
		) / 400.

		self._set_collision_signal_value(self.left_sensor)
		self._set_collision_signal_value(self.right_sensor)
		self._set_collision_signal_value(self.middle_sensor)

		return (
			self.left_sensor.signal,
			self.middle_sensor.signal,
			self.right_sensor.signal,
			self.orientation,
			-self.orientation,
		)

	def _set_collision_signal_value(self, sensor):
		if (
			sensor.position.x > self.parent.width - _PADDING or
			sensor.position.x < _PADDING or
			sensor.position.y > self.parent.height - _PADDING or
			sensor.position.y < _PADDING
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
				reward = -0.3

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


class Center(Widget):
	def __init__(self):
		Widget.__init__(self)

		self.pos = (0, 0)
		self.size = (1, 1)

		with self.canvas:
			Color(rgba=_RGBAColor.GREY)
			Ellipse(pos=self.pos, size=self.size)


class Downtown(Widget, PositionMixin):
	def __init__(self):
		self.pos = Vector(*_CHECK_POINT_OFFEST)
		self.size = (20, 20)

		Widget.__init__(self)

		with self.canvas:
			Color(rgba=_RGBAColor.GREY)
			Rectangle(pos=self.pos, size=self.size)


class Airport(Widget, PositionMixin):
	def __init__(self):
		self.pos = Vector(*Window.size) - Vector(*_CHECK_POINT_OFFEST)
		self.size = (20, 20)

		Widget.__init__(self)

		with self.canvas:
			Color(rgba=_RGBAColor.GREY)
			Rectangle(pos=self.pos, size=self.size)


class Body(Widget, PositionMixin):
	def __init__(self, pos, color):
		self.color = color
		self.pos = pos
		self.size = (20, 10)

		Widget.__init__(self)

		with self.canvas:
			Color(rgba=color)
			Rectangle(pos=self.pos, size=self.size)


class Sensor(Widget, PositionMixin):
	def __init__(self, pos, color):
		self.color = color
		self.pos = pos
		self.size = (10, 10)
		self.signal = 0.

		Widget.__init__(self)

		with self.canvas:
			Color(rgba=color)
			Ellipse(pos=self.pos, size=self.size)

	@property
	def abs_pos(self):
		# self.parent.pos + self.pos = list of 4 (x, y, z, w I guess)
		return Vector(*(self.parent.pos + self.pos)[:2])


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

			self.parent.sand[int(touch.x),int(touch.y)] = 1

	def on_touch_move(self, touch):
		if touch.button == 'left':
			touch.ud['line'].points += [touch.x, touch.y]
			self.__length += numpy.sqrt(max((touch.x - self.__last_x)**2 + (touch.y - self.__last_y)**2, 2))
			self.__n_points += 1.
			density = self.__n_points/(self.__length)
			touch.ud['line'].width = int(20 * density + 1)
			self.parent.sand[int(touch.x) - _PADDING : int(touch.x) + _PADDING, int(touch.y) - _PADDING : int(touch.y) + _PADDING] = 1
			self.__last_x = touch.x
			self.__last_y = touch.y


class Map(Widget):

	def __init__(self):
		self.cars = []
		self.size = Window.size
		self.pos = (0, 0)

		self.airport = Airport()
		self.downtown = Downtown()

		self.__plot_button = Button(
			text='plot',
			pos=(Window.width - 150, 50),
			size=(100, 50),
		)

		Widget.__init__(self)

		self.sand = numpy.zeros(
			(self.width, self.height,)
		)

	def build(self, cars_count):
		self.add_widget(self.airport)
		self.add_widget(self.downtown)
		self.add_widget(self.__plot_button)

		self.add_widget(Painter())

		for i in range(0, cars_count):
			car = Car(i, self)

			self.cars.append(car)
			self.add_widget(car)

		self.__plot_button.bind(on_release=self.__plot_learning_process)

	def update(self, dt):
		for car in self.cars:
			car.move()

	def __plot_learning_process(self, plot_button):
		for idx, car in enumerate(self.cars):
			pyplot.plot(car.scores, color=car.body.color)

		pyplot.show()

class CarApp(App):
	def __init__(self):
		App.__init__(self)

		self.__map = Map()

	def build(self):
		self.__map.build(_CARS_COUNT)

		Clock.schedule_interval(self.__map.update, 1.0/30.0)

		return self.__map


if __name__ == '__main__':
    CarApp().run()
