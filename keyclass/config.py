from yaml import load, dump
from yaml import CLoader as Loader, CDumper as Dumper

class Parser:
	def __init__(self, config_file_path='../default_config.yml', 
		     default_config_file_path='../default_config.yml'):
		"""Class to read and parse the config.yml file
		"""
		self.config_file_path = config_file_path
		with open(default_config_file_path, 'rb') as f: 
			self.default_config = load(f, Loader=Loader)

	def parse(self):
		with open(self.config_file_path, 'rb') as f: 
			self.config = load(f, Loader=Loader)
		print(self.config)

		for key, value in self.default_config.items():
			if ('target' not in key) or (key not in self.config.keys()) or (self.config[key] is None):
				self.config[key] = self.default_config[key]
				print(f'Setting the value of {key} to {self.default_config[key]}!')
		
		target_present = False
		for key in self.config.keys():
			if 'target' in key: target_present=True; break
		if not target_present: raise ValueError("Target must be present.")
		self.save_config()
		return self.config

	def save_config(self):
		with open(self.config_file_path, 'w') as f: 
			dump(self.config, f)


