import joblib
import numpy as np
from keras.models import load_model
from collections import deque

class ModelPredictor():
	#Constructor
	def __init__(self, modeltype='gb', noise_percentage=0, action_space_dim=None, \
		state_space_dim=None, markovian_order=None, model_stamp=None):
		self.action_space_dim=action_space_dim
		self.state_space_dim=state_space_dim
		self.markovian_order=markovian_order
		self.noise_percentage=noise_percentage
		self.modeltype=modeltype
		self.brain_actions=np.empty((self.action_space_dim))
		print(modeltype, ' is used as the data driven model to train brain.')
		'''
		Model stamp
		The model stamp indicates the file location depending on creation date
		it follows the following format: MONTH_DAY_HourMinuteSeconds
		using: time_stamp=time.strftime('%B_%d_%H%M%S'). If not used the script
		will check in default location.
		'''
		self.model_stamp=model_stamp

		#GRADIENT BOOST MODEL
		if modeltype=='gb':
			for i in range(0,self.state_space_dim):
				if self.model_stamp:
					filename='./models/'+self.model_stamp+'/gbmodel'+str(int(i))+'.sav'
					print("MODEL IN USE: ",filename)
					loaded_model=joblib.load(filename)
					setattr(self,'model'+str(i),loaded_model)
				else:
					try:
						filename='./sim/datadrivenmodel/models/gbmodel'+str(i)+'.sav'
						loaded_model=joblib.load(filename)
						setattr(self,'model'+str(i),loaded_model)
					except:
						print('Folder not in sim/, using current directory')
						filename='./models/gbmodel'+str(i)+'.sav'
						loaded_model=joblib.load(filename)
						setattr(self,'model'+str(i),loaded_model)

		#POLYNOMIAL MODEL
		elif modeltype=='poly':
			#Using stamp
			if self.model_stamp:
				filename='./models/'+self.model_stamp+'/polydegree.sav'
				self.polydegree=joblib.load(filename)
			else:
				try:
					self.polydegree=joblib.load('./sim/datadrivenmodel/models/polydegree.sav')
				except:
					self.polydegree=joblib.load('./models/polydegree.sav')
			print('poly degree is :', self.polydegree)
			#Using stamp
			if self.model_stamp:
				for i in range(0, self.state_space_dim):
					filename='./models/'+self.model_stamp+'/polymodel'+str(i)+'.sav'
					loaded_model=joblib.load(filename)
					setattr(self,'model'+str(i),loaded_model)
			else:
				for i in range(0, self.state_space_dim):
					try:
						filename='./sim/datadrivenmodel/models/polymodel'+str(i)+'.sav'
						loaded_model=joblib.load(filename)
						setattr(self,'model'+str(i),loaded_model)
					except:
						print('Folder not in sim/, using current directory')
						filename='./models/polymodel'+str(i)+'.sav'
						loaded_model=joblib.load(filename)
						setattr(self,'model'+str(i),loaded_model)

		#NEURONAL NETWORK MODEL
		elif modeltype=='nn':
			#Using stamp
			if self.model_stamp:
				self.model=load_model('./models/'+self.model_stamp+'/nnmodel.h5')
				self.scaler_x_set = joblib.load('./models/'+self.model_stamp+'/scaler_x_set.pkl')
				self.scaler_y_set = joblib.load('./models/'+self.model_stamp+'/scaler_y_set.pkl')
				print(self.model)

			else:
				try:
					self.model=load_model('./sim/datadrivenmodel/models/nnmodel.h5')
					self.scaler_x_set = joblib.load('./sim/datadrivenmodel/models/scaler_x_set.pkl')
					self.scaler_y_set = joblib.load('./sim/datadrivenmodel/models/scaler_y_set.pkl')
					print(self.model)
				except:
					print('Folder not in sim/, using current directory')
					self.model=load_model('./models/nnmodel.h5')
					self.scaler_x_set = joblib.load('./models/scaler_x_set.pkl')
					self.scaler_y_set = joblib.load('./models/scaler_y_set.pkl')
					print(self.model)

		#LSTM Model
		elif modeltype=='lstm':
			if self.model_stamp:
				self.model=load_model('./models/'+self.model_stamp+'/lstmmodel.h5')
				self.scaler_x_set = joblib.load('./models/'+self.model_stamp+'/scaler_x_set.pkl')
				self.scaler_y_set = joblib.load('./models/'+self.model_stamp+'/scaler_y_set.pkl')
				print(self.model)
			else:
				try:
					self.model=load_model('./sim/datadrsivenmodel/models/lstmmodel.h5')
					self.scaler_x_set = joblib.load('./sim/datadrivenmodel/models/scaler_x_set.pkl')
					self.scaler_y_set = joblib.load('./sim/datadrivenmodel/models/scaler_y_set.pkl')
					print(self.model)
					#self.action_history_to_brain=self._generate_automated_actions_name()
				except:
					print('Folder not in sim/, using current directory')
					self.model=load_model('./models/lstmmodel.h5')
					self.scaler_x_set = joblib.load('./models/scaler_x_set.pkl')
					self.scaler_y_set = joblib.load('./models/scaler_y_set.pkl')
					print(self.model)
					#self.action_history_to_brain=self._generate_automated_actions_name()
		else:
			print('ERROR: you need to specify which data driven is being used!!!')
			time.sleep(600)

	def reset_state_random(self):
		if self.modeltype=='lstm':
			self.state=deque(np.random.uniform(low=-1, high=1, size=(self.markovian_order*self.state_space_dim,)),maxlen=self.markovian_order*self.state_space_dim)
		else:
			self.state = np.random.uniform(low=-1, high=1, size=(self.state_space_dim,))
		return self.state

	def reset_state(self, config):
		if self.modeltype=='lstm':
			print("Not supported")
			exit()
		else:
			self.state = np.array([config["theta"], config["alpha"], config["theta_dot"], config["alpha_dot"]])
		return self.state

	def reset_lstm_action_history_zero(self):
		self.action_history=deque(np.zeros(shape=(self.markovian_order*self.action_space_dim,)),maxlen=self.markovian_order*self.action_space_dim)
		return self.action_history

	def predict(self, state, action=None):
		self.state=state
		if self.modeltype=='lstm':
			model_input_state=np.reshape(np.array(self.state)+np.random.uniform(low=-self.noise_percentage/100,high=self.noise_percentage/100,\
					 size=self.markovian_order*self.state_space_dim), newshape=(self.markovian_order, self.state_space_dim))
			# for key in action.keys():
			self.action_history.appendleft(action)

			model_input_actions=np.reshape(np.ravel(self.action_history), newshape=(self.markovian_order, self.action_space_dim))
			model_input=np.append(model_input_state, model_input_actions, axis=1)

			newstates=np.ravel(self.model.predict(np.array([model_input])))
			print('new state are:', newstates)
			for i in range(self.state_space_dim,0,-1):
				print('the ith state appended to the left of state', i, 'with value of: ', newstates[i-1])
				self.state=deque(self.state, maxlen=self.markovian_order*self.state_space_dim)  # I am not sure why i have to define deque again here. somewhere it becomes numpy array.
				self.state.appendleft(newstates[i-1])
		else:
			# k=0
			# for key in action.keys():
			# 	self.brain_actions[k]=(action[key])
			# 	k=k+1
			self.brain_actions=action
			model_input=np.append(self.state*(1+np.random.uniform(low=-self.noise_percentage/100,high=self.noise_percentage, size=self.state_space_dim)), self.brain_actions)

		if self.modeltype=='gb':
			self.state=[]
			for i in range(0, self.state_space_dim):
				ithmodel=getattr(self,'model'+str(i))
				self.state=np.append(self.state, ithmodel.predict(np.array([model_input])),axis=0)

		elif self.modeltype=='poly':
			self.state=[]
			#print('shape of input is: ', model_input.shape)
			model_input=self.polydegree.fit_transform([model_input])
			#print('model input after transformation is: ', model_input)
			#print('shape of input is: ', model_input.shape)
			model_input=model_input.reshape(1,-1)
			for i in range(0, self.state_space_dim):
				ithmodel=getattr(self,'model'+str(i))
				self.state=np.append(self.state, ithmodel.predict(np.array(model_input)),axis=0)

		elif self.modeltype=='nn':
			self.state=[]
			#print('model summary is:', self.model.summary())
			#print('model input after reshaping is: ', model_input)
			#print('reshape of input is: ', model_input.shape)
			model_input=self.scaler_x_set.transform([model_input])
			self.state=self.model.predict(np.array(model_input))
			self.state=self.scaler_y_set.inverse_transform(self.state)
			#print('self.state is .. :', self.state)
		elif self.modeltype=='lstm':
			pass

		self.state=np.ravel(self.state)

		return self.state
