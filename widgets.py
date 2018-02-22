from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.core.window import Window


_CHECK_POINT_OFFEST = Vector(50, 50)
_SAND_LINE_RADIUS = 10


class RGBAColor:
	RED = (1, 0, 0, 1)
	GREEN = (0, 1, 0, 1)
	BLUE = (0, 0, 1, 1)
	WHITE = (1, 1, 1, 1)
	GREY = (0.5, 0.5, 0.5, 1)


class PositionMixin:
	@property
	def position(self):
		return Vector(*self.pos)


class Center(Widget, PositionMixin):
	def __init__(self):
		Widget.__init__(self)

		self.pos = (0, 0)
		self.size = (1, 1)

		with self.canvas:
			Color(*RGBAColor.GREY)
			Ellipse(pos=self.pos, size=self.size)


class Downtown(Widget, PositionMixin):
	def __init__(self):
		self.pos = Vector(*_CHECK_POINT_OFFEST)
		self.size = (20, 20)

		Widget.__init__(self)

		with self.canvas:
			Color(*RGBAColor.GREY)
			Rectangle(pos=self.pos, size=self.size)

	def __repr__(self):
		return "Downtown"


class Airport(Widget, PositionMixin):
	def __init__(self):
		self.pos = Vector(*Window.size) - Vector(*_CHECK_POINT_OFFEST)
		self.size = (20, 20)

		Widget.__init__(self)

		with self.canvas:
			Color(*RGBAColor.GREY)
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


class Sensor(Widget, PositionMixin):
	def __init__(self, pos, color, rotation):
		self.color = color
		self.pos = pos
		self.size = (10, 10)
		self.signal = 0.0

		self._rotation = rotation

		Widget.__init__(self)

		with self.canvas:
			Color(*self.color)
			Ellipse(pos=self.pos, size=self.size)

	@property
	def abs_pos(self):
		return self.parent.position + self.position.rotate(self._rotation.angle)


class Painter(Widget):
	def __init__(self):
		Widget.__init__(self)

	def on_touch_down(self, touch):
		with self.canvas:
			Color(0.8,0.7,0)
			touch.ud['line'] = Line(points = (touch.x, touch.y), width = _SAND_LINE_RADIUS * 2)

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


