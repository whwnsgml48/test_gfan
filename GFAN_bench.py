import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 

from Model.attention_layer import StandardAttention, Dense_Attention
from utils.preprocessing_bench import load_bench_iris


if __name__ == '__main__':
	path = './Results/Benchmark'
	now_benchmark = sys.argv[1]# iris,...
	now_benchmark = 'iris'

	# load datatsets
	if now_benchmark = 'iris':
		Xs, Adj, Y_train, y_test, idx_train = load_bench_iris()
	else:
		print('what?')
		sys.exit()
		# 데이터 셋 추가시 util에서 load하는 function 만든 후 Xs, Adj, Y_train, y_test, idx_train 반환 

	#feature_importance 
	delta = np.ones((Xs[0].shape[0],Xs[0].shape[1]))

	# parameter setting
	classes = Y_train.shape[1] 
	In_epochs = 100  
	Out_epochs = 2

	seed1 = 10 
	seed2 = 11 
	seed3 = 12 
	seed4 = 13

	losses_ = [] 
	Attn_ = []

	# 
	for k in range(len(Xs)):     
		X = np.copy(Xs[k,:,:]) .astype(np.float32)  
		
		# Call model     
		model = Dense_Attention(A=Adj, seed1=seed1, seed2=seed2, seed3=seed3, seed4=seed4, num_outputs=classes) 
	    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),                   
	    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])     
	    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)       

	    # Train the model     
	    history = model.fit(X, Y_train, batch_size=X.shape[0], epochs = In_epochs, callbacks=[es_callback], validation_data=(X,y_test), sample_weight=idx_train)     

	    if k == 0:         
	    	path0 = path + '/logs_' + str(now_benchmark)         
	    	model.save(path0, save_format='tf') 
	    	model_loss = history.history['loss']     

	    # Feature Importance (Datapoint wise loss)     
	    losses = np.empty((X.shape[0],1))     
	    pred = model.predict(X, batch_size = X.shape[0])     
	    cce = tf.keras.losses.CategoricalCrossentropy()     
	    for i in range(X.shape[0]):         
	    	losses[i,0] = cce(pred[i], Y_train[i]).numpy()     
	    losses_.append(losses)

	# 
	H = np.copy(Xs)     
	for jj in range(Out_epochs):         
		losses_ = []         
		Attn_ = []         
		for p in range(len(H)):             
			H[p,:,:] *= delta

		for k in range(len(H)):             
			X = H[k, :, :]  .astype(np.float32)

			# Call model             
			path10 = path + '/logs_' + str(now_benchmark)
			new_model = tf.keras.models.load_model(path10)  

			new_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),                               
							  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])  

			es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)  

			# Train the model             
			history = new_model.fit(X, Y_train, batch_size=X.shape[0], epochs=In_epochs, callbacks=[es_callback], 
			                        validation_data=(X, y_test), sample_weight=idx_train)             
			# Save the model             
			if k == 0:                 
				path0 = path + '/logs_' + str(now_benchmark)
				new_model.save(path0, save_format='tf')                 
				model_loss = history.history['loss']  

			# Feature Importance (Datapoint wise loss)             
			losses = np.empty((X.shape[0], 1))             
			pred = new_model.predict(X, batch_size=X.shape[0])             
			cce = tf.keras.losses.CategoricalCrossentropy()             

			for i in range(X.shape[0]):                 
				losses[i, 0] = cce(pred[i], Y_train[i]).numpy()             
			losses_.append(losses)          

		FI = []         
		for i in range(len(losses_) - 1):             
			FI.append(losses_[i + 1] / losses_[0])         
		for idx, val in enumerate(nonzero_idx):             
			delta[:, val] = FI[idx]          
		#path1 = path + "/Attn_"+str(now_benchmark)+"_GFAN_" + str(jj) + "_Out.csv"         
		path2 = path + "/FI_"+str(now_benchmark)+"_GFAN_" + str(jj) + "_Out.csv"         
		path3 = path + "/loss_"+str(now_benchmark)+"_GFAN_" + str(jj) + "_Out.csv"         
		
		#np.savetxt(path1, Attn_[0], delimiter=',')         
		np.savetxt(path2, delta, delimiter=',')         
		np.savetxt(path3, np.asarray(model_loss), delimiter=',')      
