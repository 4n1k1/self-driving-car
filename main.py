#!/usr/bin/python

import time
import numpy as np
from matplotlib import pyplot as plt

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

from ai import Brain

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

class Car(Widget):
    
	__INPUT_SIZE = 5  # [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
	__ROTATIONS = [0, 20, -20]
	__DISCOUNT_FACTOR = 0.9

	angle = NumericProperty(0)
	rotation = NumericProperty(0)
	velocity_x = NumericProperty(0)
	velocity_y = NumericProperty(0)
	velocity = ReferenceListProperty(velocity_x, velocity_y)
	sensor1_x = NumericProperty(0)
	sensor1_y = NumericProperty(0)
	sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
	sensor2_x = NumericProperty(0)
	sensor2_y = NumericProperty(0)
	sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
	sensor3_x = NumericProperty(0)
	sensor3_y = NumericProperty(0)
	sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
	signal1 = NumericProperty(0)
	signal2 = NumericProperty(0)
	signal3 = NumericProperty(0)

	def __init__(self):
		Widget.__init__(self)

		self.__brain = Brain(
			self.__INPUT_SIZE,
			self.__ROTATIONS,
			self.__DISCOUNT_FACTOR,
		)

		self.__scores = []

	def move(self, rotation):
		self.pos = Vector(*self.velocity) + self.pos
		self.rotation = rotation
		self.angle = self.angle + self.rotation

		self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
		self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
		self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos

		self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
		self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
		self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.

		if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
			self.signal1 = 1.
		if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
			self.signal2 = 1.
		if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
			self.signal3 = 1.


_MAP_WIDTH = 20
_MAP_HEIGHT = 20

_DESTINATION_X = 20
_DESTINATION_Y = 0


class Map(Widget):

	def __init__(self):
		Widget.__init__(self)

		self.__cars = []

		for i in range(1, _NUMBER_OF_CARS):
			self.__cars.append(ObjectProperty(None))

		self.__sand = np.zeros(
			(self.width, self.height,)
		)

		self.__goal_x = _DESTINATION_X
		self.__goal_y = _DESTINATION_Y

	def save(self):
		for car in self.__cars:
			car.save()

	def serve_car(self):
		self.__car.center = self.center
		self.__car.velocity = Vector(6, 0)

	def clear(self):
		self.__sand = np.zeros((self.width, self.height,))

	def update(self, dt):
		xx = goal_x - self.car.x
		yy = goal_y - self.car.y

		orientation = Vector(*self.car.velocity).angle((xx,yy))/180.

		last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]

		action = brain.update(last_reward, last_signal)

		scores.append(brain.score())
		rotation = action2rotation[action]
		self.car.move(rotation)
		distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
		self.ball1.pos = self.car.sensor1
		self.ball2.pos = self.car.sensor2
		self.ball3.pos = self.car.sensor3

		if sand[int(self.car.x),int(self.car.y)] > 0:
			self.car.velocity = Vector(1, 0).rotate(self.car.angle)
			last_reward = -0.5
		else:
			self.car.velocity = Vector(6, 0).rotate(self.car.angle)
			last_reward = -0.2
			if distance < last_distance:
				last_reward = 0.1

		if self.car.x < 10:
			self.car.x = 10
			last_reward = -1
		if self.car.x > self.width - 10:
			self.car.x = self.width - 10
			last_reward = -1
		if self.car.y < 10:
			self.car.y = 10
			last_reward = -1
		if self.car.y > self.height - 10:
			self.car.y = self.height - 10
			last_reward = -1

		if distance < 100:
			goal_x = self.width-goal_x
			goal_y = self.height-goal_y

		last_distance = distance


class MyPaintWidget(Widget):

	def __init__(self, *args, **kwargs):
		Widget.__init__(self, args, kwargs)

		self.__last_x = 0
		self.__last_y = 0
		self.__length = 0
		self.__n_points = 0

	def on_touch_down(self, touch):
		with self.canvas:
			Color(0.8,0.7,0)
			touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)

			self.__last_x = int(touch.x)
			self.__last_y = int(touch.y)
			self.__n_points = 0
			self.__length = 0

			sand[int(touch.x),int(touch.y)] = 1

	def on_touch_move(self, touch):
		if touch.button == 'left':
			touch.ud['line'].points += [touch.x, touch.y]
			length += np.sqrt(max((x - self.__last_x)**2 + (y - self.__last_y)**2, 2))
			self.__n_points += 1.
			density = self.__n_points/(length)
			touch.ud['line'].width = int(20 * density + 1)
			sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
			self.__last_x = x
			self.__last_y = y

# Adding the API Buttons (clear, save and load)
class CarApp(App):

	def __init__(self):
		self.__map = Map()
		self.__painter = MyPaintWidget()

	def build(self):
		self.__map.serve_car()

		Clock.schedule_interval(self.__map.update, 1.0/60.0)

		clearbtn = Button(text = 'clear')
		savebtn = Button(text = 'save', pos = (parent.width, 0))
		loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))

		clearbtn.bind(on_release=self.__clear_canvas)
		savebtn.bind(on_release=self.__save)
		loadbtn.bind(on_release=self.__load)

		self.__map.add_widget(self.__painter)
		self.__map.add_widget(clearbtn)
		self.__map.add_widget(savebtn)
		self.__map.add_widget(loadbtn)

		return self.__map

	def __clear_canvas(self, obj):
		self.__map.clear()
		self.__painter.canvas.clear()

	def __save(self, obj):
		self.__map.save()

		plt.plot(scores)
		plt.show()

	def __load(self, obj):
		brain.load()


if __name__ == '__main__':
    CarApp().run()
