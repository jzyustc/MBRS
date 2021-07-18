'''
class to load settings config

By jzyustc, 2020/12/16

'''
import json


class JsonConfig:

	def __init__(self, ):
		self.__json__ = None

	def load_json_file(self, path: str):
		with open(path, "r") as file:
			self.__json__ = json.load(file)
			file.close()

		self.set_items()

	def load_json(self, json: dict):
		self.__json__ = json

		self.set_items()

	def set_items(self):

		for key in self.__json__:
			self.__setattr__(key, self.__json__[key])

	def get_items(self):
		items = []
		for key in self.__json__:
			items.append((key, self.__json__[key]))
		return items
