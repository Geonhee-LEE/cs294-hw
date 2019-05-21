import pickle, tensorflow as tf, tf_util, numpy as np

'''
Hopper-v2의 경우로 주석처리, action = 3(thigh_joint, leg_joint, foot_joint)
'''

def load_policy(filename):
    '''
    loading and building expert policy
    '''
    print('################ Env: ', filename, '###################')
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
        #print(type(data)) #<class 'dict'>

    # assert len(data.keys()) == 2
    nonlin_type = data['nonlin_type']
    #print(nonlin_type) # tanh
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]
    #print(policy_type) # GaussianPolicy

    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type) #assert type(t) is int, '정수 아닌 값이 있네'
    
    policy_params = data[policy_type]
    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}
    ''' 집합 자료형
    >>> s1 = set([1,2,3])
    >>> s1
    {1, 2, 3}
    '''

    # Keep track of input and output dims (i.e. observation and action dims) for the user

    # Encapsulation
    def build_policy(obs_bo):
        def read_layer(layer_data):
            '''
            Extract Weight, bias from layer <class 'dict'>
            '''
            assert list(layer_data.keys()) == ['AffineLayer']
            assert sorted(layer_data['AffineLayer'].keys()) == ['W', 'b']
            return layer_data['AffineLayer']['W'].astype(np.float32), layer_data['AffineLayer']['b'].astype(np.float32)
            '''
            numpy.ndarray.astype
                Copy of the array, cast to a specified type.
            '''

        def apply_nonlin(x):
            '''
            Apply the nonlinear activation function such as leack relu, tanh
            '''
            if nonlin_type == 'lrelu':
                return tf_util.lrelu(x, leak=.01) # openai/imitation nn.py:233
            elif nonlin_type == 'tanh':
                return tf.tanh(x)
            else:
                raise NotImplementedError(nonlin_type)

        # Build the policy. First, observation normalization.
        assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D'] # <class 'numpy.ndarray'>
        obsnorm_meansq = policy_params['obsnorm']['Standardizer']['meansq_1_D'] # <class 'numpy.ndarray'>
        obsnorm_stdev = np.sqrt(np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean))) # <class 'numpy.ndarray'>, standard deviation = \sqrt{E( X^2 ) - ( E(X) )^2}
        print('observation mean, standard deviation shape: ', obsnorm_mean.shape, obsnorm_stdev.shape) #(1, 11)
        
        normedobs_bo = (obs_bo - obsnorm_mean) / (obsnorm_stdev + 1e-6) # 1e-6 constant from Standardizer class in nn.py:409 in openai/imitation
        ''' Nomalized observation _ behavior observation (Standard score \frac{X-\mu}{\sigma})
        obs_bo          = X
        obsnorm_mean    = /mu
        obsnorm_stdev   = \sigma
        ----------------------------
        normedobs_bo   = normalized data
        '''

        curr_activations_bd = normedobs_bo

        # Hidden layers next
        assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']

        layer_params = policy_params['hidden']['FeedforwardNet'] # < class 'dict'>, layer_0, layer_2        
        
        # 2 layers
        for layer_name in sorted(layer_params.keys()): # <class 'str'>, , layer_name = layer_0, layer_2
            '''
            Pass the layers given from expert, 
            '''
            layer_data = layer_params[layer_name]  # < class 'dict'>, layer_data = {'W', 'b'}
            W, b = read_layer(layer_data) # layer_0:  (11, 64) (1, 64), layer_2: (64, 64) (1, 64)
            print(W.shape, b.shape)
            curr_activations_bd = apply_nonlin(tf.matmul(curr_activations_bd, W) + b) # current activation behavior data + nonlinear activation funtion

        print('----end---')
        # Output layer, 1 layer
        W, b = read_layer(policy_params['out']) # (64, 3) (1, 3)
        # print(W.shape, b.shape)
        
        output_bo = tf.matmul(curr_activations_bd, W) + b # (?, 3), ?은 위의 과정에서 (1, 11)과 브로팅캐스팅 진행
        #print(output_bo.shape)
        
        return output_bo # Output behavior output

    #we create pairs of <observation, action>
    obs_bo = tf.placeholder(tf.float32, [None, None])  # <class 'tensorflow.python.framework.ops.Tensor'>, Tensor("Placeholder:0", shape=(?, ?), dtype=float32) 
    a_ba = build_policy(obs_bo) # Output behavior, <class 'tensorflow.python.framework.ops.Tensor'>, Tensor("add_2:0", shape=(?, 3), dtype=float32)
    
    policy_fn = tf_util.function([obs_bo], a_ba) # <class 'function'>, 
    '''
    function(inputs, outputs, updates=None, givens=None)
    [obs_bo]: list, [<tf.Tensor 'Placeholder:0' shape=(?, ?) dtype=float32>]
    a_ba: <class 'tensorflow.python.framework.ops.Tensor'>, Tensor("add_2:0", shape=(?, 3), dtype=float32)
    '''

    return policy_fn