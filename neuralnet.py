
def make_params(shape_list): # shape_list = [784, 100, 10]のように層ごとにニューロンの数を配列にしたものを入力する
    weight_list = []
    bias_list = []
    for i in range(len(shape_list)-1):
        weight = np.random.randn(shape_list[i], shape_list[i+1]) # 標準正規分布に従った乱数を初期値とする
        bias = np.ones(shape_list[i+1])/10.0 # 初期値はすべて0.1にする
        weight_list.append(weight)
        bias_list.append(bias)
    return weight_list, bias_list

def sigmoid(x): # シグモイド関数
    return 1/(1+np.exp(-x))

def inner_product(X, w, b): # ここは内積とバイアスを足し合わせる
    return np.dot(X,w)+ b

def activation(X, w, b):
    return sigmoid(inner_product(X, w, b))

def calculate(X, w_list, b_list, t): # 層ごとの計算結果を格納した配列を返す。
    val_list = {}
    a_1 = inner_product(X, w_list[0], b_list[0]) # (N, 1000)
    y_1 = sigmoid(a_1) # (N, 100)
    a_2 = inner_product(y_1, w_list[1], b_list[1]) # (N, 10)
    y_2 = sigmoid(a_2) # これが本来は求めたい値。(N,10)
    y_2 /= np.sum(y_2, axis=1, keepdims=True) # ここで簡単な正規化をはさむ
    S = 1/(2*len(y_2))*(y_2 - t)**2
    L = np.sum(S)
    val_list['a_1'] = a_1
    val_list['y_1'] = y_1
    val_list['a_2'] = a_2
    val_list['y_2'] = y_2
    val_list['S'] = S
    val_list['L'] = L
    return val_list

def predict(X, w_list, b_list, t): # ここで予想を行う。
    val_list = calculate(X, w_list, b_list, t)
    y_2 = val_list['y_2']
    result = np.zeros_like(y_2)
    for i in range(y_2.shape[0]): # サンプル数にあたる
        result[i, np.argmax(y_2[i])] = 1
    return result

def accuracy(X, w_list, b_list, t):
    pre = predict(X, w_list, b_list, t)
    result = np.where(np.argmax(t, axis=1)==np.argmax(pre, axis=1), 1, 0)
    acc = np.mean(result)
    return acc
def loss(X, w_list, b_list, t):
    L = calculate(X, w_list, b_list, t)['L']
    return L

def update(X, w_list, b_list, t, eta): # etaは学習率。ここでパラメータの更新を行う
    val_list = {}
    val_list = calculate(X, w_list, b_list, t)
    a_1 = val_list['a_1']
    y_1 = val_list['y_1']
    a_2 = val_list['a_2']
    y_2 = val_list['y_2']
    S = val_list['S']
    L = val_list['L']
    dL_dS = 1.0
    dS_dy_2 = 1/X.shape[0]*(y_2 - t)
    dy_2_da_2 = y_2*(1.0 - y_2)
    da_2_dw_2 = np.transpose(y_1)
    da_2_db_2 = 1.0
    da_2_dy_1 = np.transpose(w_list[1])
    dy_1_da_1 = y_1 * (1 - y_1)
    da_1_dw_1 = np.transpose(X)
    da_1_db_1 = 1.0
    # ここからパラメータの更新を行っていく。
    dL_da_2 =  dL_dS * dS_dy_2 * dy_2_da_2
    b_list[1] -= eta*np.sum(dL_da_2 * da_2_db_2, axis=0)
    w_list[1] -= eta*np.dot(da_2_dw_2, dL_da_2)
    dL_dy_1 = np.dot(dL_da_2, da_2_dy_1)
    dL_da_1 = dL_dy_1 * dy_1_da_1
    b_list[0] -= eta*np.sum(dL_da_1 * da_1_db_1, axis=0)
    w_list[0] -= eta*np.dot(da_1_dw_1, dL_da_1)
    return w_list, b_list

