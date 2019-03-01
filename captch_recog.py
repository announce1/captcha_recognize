from captcha.image import ImageCaptcha
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import os

# train_data_dir = r'F:\Tensorflow_yzm\generate' # 根据实际情况替换
# test_data_dir  = r'F:\Tensorflow_yzm\test'
# train_data_dir = r'F:\train' # 根据实际情况替换
# test_data_dir  = r'F:\test'

## 用于生成验证码的字符集
CHAR_SET = ['0','1','2','3','4','5','6','7','8','9']

CHAR_SET_LEN = len(CHAR_SET) ## 字符集的长度

MAX_CAPTCHA = 4              ## 验证码的长度，每个验证码由4个数字组成

img_height = 60              ## 图片高
img_width  = 160             ## 图片宽

## 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img

## 随机生成字符
def random_captcha_text(char_set=CHAR_SET, captcha_size=MAX_CAPTCHA):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text
    
## 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    
    captcha = image.generate(captcha_text)
    #image.write(captcha_text, captcha_text + '.jpg')  # 写到文件
    
    captcha_image = Image.open(captcha)
    captcha_image_np = np.array(captcha_image)
    captcha_image_np = convert2gray(captcha_image_np)
    
    return captcha_text, captcha_image_np
    
##################################
## 描述：生成一个batch数量的训练数据集
## batch_size：一次生成的batch的数量
## return：
##     x_data：图片数据集  x_data.shape = (64, 60, 160, 1)
##     y_data：标签集      y_data.shape = (64, 4)
##################################
def gen_one_batch(batch_size=32):

    x_data = []
    y_data = []
    
    for i in range(batch_size):
        captcha_text, captcha_image_np = gen_captcha_text_and_image() ## captcha_image_np.shape = (60,160)
        assert captcha_image_np.shape == (60, 160)
        captcha_image_np = np.expand_dims(captcha_image_np, 2)
        x_data.append(captcha_image_np)
        y_data.append(np.array(list(captcha_text)).astype(np.int32))
    x_data = np.array(x_data).astype(np.float) ## x_data.shape = (64, 60, 160, 1)
    y_data = np.array(list(y_data))            ## y_data.shape = (64, 4)
    return x_data, y_data

X = tf.placeholder(tf.float32, name="input") ## 亦即 X = x_data , so X.shape = (64, 24, 60, 1)
Y = tf.placeholder(tf.int32)                 ## 亦即 Y = y_data , so Y.shape = (64, 4)
keep_prob = tf.placeholder(tf.float32)
y_one_hot = tf.one_hot(Y, 10, 1, 0)          ## y_one_hot.shape = (batch_size , 4 , 10)
y_one_hot = tf.cast(y_one_hot, tf.float32)   ## tf.cast()类型转换

# keep_prob = 1.0
def net(w_alpha=0.01, b_alpha=0.1):
    '''
    网络部分，三层卷积层，一个全连接层
    :param w_alpha:
    :param b_alpha:
    :return: 网络输出，Tensor格式
    '''
    conv2d_size = 3      ## 卷积核大小
    featuremap_num1 = 32 ## 卷积层1输出的featuremap的数量
    featuremap_num2 = 64 ## 卷积层2输出的featuremap的数量
    featuremap_num3 = 64 ## 卷积层3输出的featuremap的数量
    
    strides_conv2d1 = [1, 1, 1, 1] ##卷积层1的卷积步长
    strides_conv2d2 = [1, 1, 1, 1] ##卷积层2的卷积步长
    strides_conv2d3 = [1, 1, 1, 1] ##卷积层3的卷积步长
    
    strides_pool1   = [1, 2, 2, 1] ## 卷积层1的池化步长
    strides_pool2   = [1, 2, 2, 1] ## 卷积层2的池化步长
    strides_pool3   = [1, 2, 2, 1] ## 卷积层3的池化步长
    
    ksize_pool1     = [1, 2, 2, 1] ## 卷积层1的池化size
    ksize_pool2     = [1, 2, 2, 1] ## 卷积层2的池化size
    ksize_pool3     = [1, 2, 2, 1] ## 卷积层3的池化size
    
    neuron_num      = 1024         ## 神经元数量
    
    FC_dim1         = float(img_height)/(strides_pool1[1]*strides_pool2[1]*strides_pool3[1])
    FC_dim2         = float(img_width) /(strides_pool1[2]*strides_pool2[2]*strides_pool3[2])
    FC_dim          = int(round(FC_dim1) * round(FC_dim2) * featuremap_num3)
    
    ## -1代表先不考虑输入的图片有多少张，1是channel的数量 
    x_reshape = tf.reshape(X,shape = [-1, img_height, img_width, 1])
    
    ## 构建卷积层1 
    ## tf.Variable()定义变量
    ## tf.random_normal()生成N维服从正太分布的数据
    w_c1 = tf.Variable(w_alpha * tf.random_normal([conv2d_size, conv2d_size, 1, featuremap_num1])) # 卷积核3*3，1个channel，16个卷积核，形成16个featuremap
    b_c1 = tf.Variable(b_alpha * tf.random_normal([featuremap_num1])) # 16个featuremap的偏置
    ## tf.nn.relu()激活函数，激活函数是用来加入非线性因素的，提高神经网络对模型的表达能力，解决线性模型所不能解决的问题。
    ## tf.nn.bias_add(value, bias, data_format=None, name=None),将偏差项bias加到values上
    ## tf.nn.conv2d()卷积计算的核心函数,讲解请参考https://www.cnblogs.com/qggg/p/6832342.html
    ## padding参数中SAME代表给边界加上Padding让卷积的输出和输入保持相同的尺寸
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_reshape, w_c1, strides=strides_conv2d1, padding='SAME'), b_c1))
    ## tf.nn.max_pool()最大值池化
    ## 经过tf.nn.max_pool(strides=[1, 2, 2, 1])后，feature_map在各个维度上都变为一半
    conv1 = tf.nn.max_pool(conv1, ksize=ksize_pool1, strides=strides_pool1, padding='SAME')
    ## tf.nn.dropout()此函数是为了防止在训练中过拟合的操作，将训练输出按一定规则进行变换
    conv1 = tf.nn.dropout(conv1, keep_prob)

    ## 构建卷积层2
    w_c2 = tf.Variable(w_alpha * tf.random_normal([conv2d_size, conv2d_size, featuremap_num1, featuremap_num2])) # 注意这里channel值是16
    b_c2 = tf.Variable(b_alpha * tf.random_normal([featuremap_num2]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=strides_conv2d1, padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=ksize_pool2, strides=strides_pool2, padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    
    ## 构建卷积层3
    w_c3 = tf.Variable(w_alpha * tf.random_normal([conv2d_size, conv2d_size, featuremap_num2, featuremap_num3]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([featuremap_num3]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=strides_conv2d1, padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=ksize_pool3, strides=strides_pool3, padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    ## 构建全连接层，这个全连接层的输出才是最后要提取的特征
    # Fully connected layer
    # 随机生成权重
    ## https://stackoverflow.com/questions/43010339/python-tensorflow-input-to-reshape-is-a-tensor-with-92416-values-but-the-re
    # w_d = tf.Variable(w_alpha * tf.random_normal([3 * 8 * 64, 128]))
    # w_d = tf.Variable(w_alpha * tf.random_normal([163840, 128]))   # 128个神经元
    w_d = tf.Variable(w_alpha * tf.random_normal([FC_dim, neuron_num]))   # 1024个神经元
    # 随机生成偏置
    b_d = tf.Variable(b_alpha * tf.random_normal([neuron_num]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    ## 最后要提取的特征
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))

    ## 输出层
    w_out = tf.Variable(w_alpha * tf.random_normal([neuron_num, 4 * 10]))  # 40个神经元
    b_out = tf.Variable(b_alpha * tf.random_normal([4 * 10]))
    out = tf.add(tf.matmul(dense, w_out), b_out) ## out.shape = (64,40)
    ## 最后一个全连接层的输出维度，在设计时是和训练样本的类别数一致的 , out.shape == y_one_hot.shape == = (batch_size , 4 , 10)
    out = tf.reshape(out, (-1, 4, 10))
    return out
cnn = net()
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cnn, labels=y_one_hot))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cnn, labels=y_one_hot))
# optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

def train():
    
    batch_size_train = 64  ## 一个训练batch的数量
    batch_size_test  = 100 ## 一个测试batch的数量

    print('开始执行训练')

    predict = net()
    max_idx_p = tf.argmax(predict, 2)  ## 通过argmax返回的index可得出识别的图片的字符值
    max_idx_l = tf.argmax(tf.reshape(y_one_hot, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        step = 0
        tf.global_variables_initializer().run()
        while True:
            x_data, y_data = gen_one_batch(batch_size_train)
            loss_, cnn_, y_one_hot_, optimizer_ = sess.run([loss, cnn, y_one_hot, optimizer],
                                                           feed_dict={Y: y_data, X: x_data, keep_prob: 0.75})
            print('step: %4d, loss: %.4f' % (step, loss_))
                
            # 每100 step计算一次准确率
            if 0 == (step % 100):
                x_data, y_data = gen_one_batch(batch_size_test)
                acc = sess.run(accuracy, feed_dict={X: x_data, Y: y_data, keep_prob: 1.0})
                print('准确率计算:step: %4d, accuracy: %.4f' % (step, acc))
                if acc > 0.99:
                    saver.save(sess, "tmp/", global_step=step)
                    print("训练完成，模型保存成功！")
                    break
            step += 1
def gen_test_data():
    x_data = []
    y_data = []
    for parent, dirnames, filenames in os.walk(test_data_dir, followlinks=True):
        for filename in filenames:
            gif_file_path = os.path.join(parent, filename)
            if gif_file_path.endswith('.gif'):
                captcha_image = Image.open(gif_file_path)
                captcha_image_np = np.array(captcha_image)
                assert captcha_image_np.shape == (60, 160)
                captcha_image_np = np.expand_dims(captcha_image_np, 2).astype(np.float32)
                x_data.append(captcha_image_np)
                y_data.append(filename.split('.')[0])
    return x_data, y_data
def test():
    if not os.path.exists(test_data_dir):
        raise RuntimeError('测试数据目录不存在，请检查"%s"参数' % 'test_data_dir')
    if tf.train.latest_checkpoint('tmp/') is None:
        raise RuntimeError('未找到模型文件，请先执行训练！')
    print('%s' % '开始执行测试')
    x, y = gen_test_data()
    print('测试目录文件数量：%d' % len(x))
    saver = tf.train.Saver()
    sum = 0
    correct = 0
    error = 0
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('tmp/'))
        for i, image in enumerate(x):
            answer = y[i]
            image = image.reshape((1, img_height , img_width, 1))
            cnn_out = sess.run(cnn, feed_dict={X: image, keep_prob: 1})
            # print(cnn_out)
            cnn_out = cnn_out[0]
            predict_vector = np.argmax(cnn_out, 1)
            predict = ''
            for c in predict_vector:
                predict += str(c)
            print('预测：%s，答案：%s，判定：%s' % (predict, answer, "√" if predict == answer else "×"))
            sum += 1
            if predict == answer:
                correct += 1
            else:
                error += 1
    print("总数：%d，正确：%d，错误：%d" % (sum, correct, error))
if __name__=='__main__':
    # 训练
    train()
    # 测试
    test()
