import tensorflow as tf

from tensorflow.keras.layers import Layer  


class StandardAttention(Layer): 
	 def __init__(self, num_outputs, A, seed2, seed3, seed4):     
		super(StandardAttention, self).__init__()     
		self.num_outputs = num_outputs     
		self.A = A     
		self.sid2 = seed2     
		self.sid3 = seed3     
		self.sid4 = seed4 

	def build(self, input_shape):     
		#Layer kernel    
		self.w = self.add_weight(name='w',shape = (input_shape[-1], self.num_outputs), 
		                         initializer = tf.keras.initializers.GlorotUniform(seed=self.sid2), 
		                         trainable = True)     

		#Layer bias     
		self.b = self.add_weight(name='b',shape=(self.num_outputs,), 
	                             initializer='zeros',                              
	                             trainable = True)     

	    #Attention kernel     
	    self.a_self = self.add_weight(name='a_self',shape=(self.num_outputs, 1), 
	                                  initializer = tf.keras.initializers.GlorotUniform(seed=self.sid3), 
	                                  trainable = True)     
	    self.a_neighbor = self.add_weight(name='a_neighbor',shape=(self.num_outputs, 1), 
	                                      initializer=tf.keras.initializers.GlorotUniform(seed=self.sid4), 
	                                      trainable=True)     

	    #self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])  
	def call(self, inputs):     
		# Compute inputs to attention network     
		features = tf.matmul(inputs, self.w) + self.b  # (N x F')      
		# Compute feature combinations     
		# Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]     
		attn_for_self = tf.matmul(features, self.a_self)  # (N x 1), [a_1]^T [Wh_i]     
		attn_for_neighs = tf.matmul(features, self.a_neighbor)  # (N x 1), [a_2]^T [Wh_j]  

		# Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]     
		attn_coef = attn_for_self + tf.transpose(attn_for_neighs)  # (N x N) via broadcasting      

		# Add nonlinearty     
		attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)      

		# Mask values before activation (Vaswani et al., 2017)     
		mask = -10e9 * (1.0 - self.A)     
		attn_coef += mask      

		# Apply softmax to get attention coefficients     
		attn_coef = tf.nn.softmax(attn_coef)  # (N x N)      

		# Apply dropout to features and attention coefficients     
		##dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)     
		##dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F')      

		# Linear combination with neighbors' features     
		node_features = tf.matmul(attn_coef, features)  # (N x F')      

		#if self.use_bias:     
		#    node_features = K.bias_add(node_features, self.biases[head])      

		return tf.nn.softmax(node_features) 



class Dense_Attention(tf.keras.Model):      
	def __init__(self, A, seed1, seed2, seed3, seed4, input_shape=(Data_file.shape[1],), num_outputs=3): 
	        super(Dense_Attention, self).__init__()         
	        self.num_outputs=num_outputs         
	        self.A=A         
	        self.sid1=seed1         
	        self.sid2=seed2         
	        self.sid3=seed3         
	        self.sid4=seed4         
	        self.dense = Dense(10, input_shape=input_shape, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.sid1), bias_initializer='zeros')         
	        self.standardattention = StandardAttention(num_outputs, A, self.sid2, self.sid3, self.sid4)      

	def call(self, inputs):         
		x = self.dense(inputs)         
		return self.standardattention(x)
