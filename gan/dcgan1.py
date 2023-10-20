import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

num_features = 784 #28*28*1，输入的特征数

# 训练参数.
lr_generator = 0.0002
lr_discriminator = 0.0002
training_steps = 2000
batch_size = 128
display_step = 100

# 初始化随机向量维度
noise_dim = 100


# 加载数据
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 类型转换
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# 特征归一化
x_train, x_test = x_train / 255., x_test / 255.

# 制作batch数据
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(10000).batch(batch_size).prefetch(1)


# 创建生成器模型
# 输入: 随机向量, Output: 生成的图像数据
class Generator(Model):
    # 用到的层
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = layers.Dense(7 * 7 * 128)
        self.bn1 = layers.BatchNormalization()
        self.conv2tr1 = layers.Conv2DTranspose(64, 5, strides=2, padding='SAME')
        self.bn2 = layers.BatchNormalization()
        self.conv2tr2 = layers.Conv2DTranspose(1, 5, strides=2, padding='SAME')

    # 前向传播计算
    def call(self, x, is_training=False):
        x = self.fc1(x) # Input:S x 100, Output:S x 6272 (7x7x128), Weights: (100+1)x6272
        x = self.bn1(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # 转换成4-D图像数据: (batch, height, width, channels)
        # (batch, 7, 7, 128)
        x = tf.reshape(x, shape=[-1, 7, 7, 128]) # S x 6272 -> S x 7 x 7 x 128
        # (batch, 14, 14, 64)
        x = self.conv2tr1(x) # S x 7 x 7 x 128 -> S x 14 x 14 x 64
        x = self.bn2(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # 还原成(batch, 28, 28, 1)
        x = self.conv2tr2(x) # S x 14 x 14 x 64 -> S x 28 x 28 x 1
        x = tf.nn.tanh(x) #
        return x # S x 28 x 28 x 1

#判别器模型
class Discriminator(Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, 5, strides=2, padding='SAME')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(128, 5, strides=2, padding='SAME')
        self.bn2 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024)
        self.bn3 = layers.BatchNormalization()
        self.fc2 = layers.Dense(2)

    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.bn1(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn3(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        return self.fc2(x) # S x 2

# 创建网络模型
generator = Generator()
discriminator = Discriminator()


# 损失函数
def generator_loss(reconstructed_image):
    gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=reconstructed_image, labels=tf.ones([batch_size], dtype=tf.int32)))
    return gen_loss

def discriminator_loss(disc_fake, disc_real):
    disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)))
    disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)))
    return disc_loss_real + disc_loss_fake

# 优化器
optimizer_gen = tf.optimizers.Adam(learning_rate=lr_generator)#, beta_1=0.5, beta_2=0.999)
optimizer_disc = tf.optimizers.Adam(learning_rate=lr_discriminator)#, beta_1=0.5, beta_2=0.999)


def run_optimization(real_images):
    # 将特征处理成 [-1, 1]
    real_images = real_images * 2. - 1.

    # 随机产生噪音数据
    noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)

    with tf.GradientTape() as g:
        fake_images = generator(noise, is_training=True)
        disc_fake = discriminator(fake_images, is_training=True)
        disc_real = discriminator(real_images, is_training=True)

        disc_loss = discriminator_loss(disc_fake, disc_real)

    # 判别器优化
    gradients_disc = g.gradient(disc_loss, discriminator.trainable_variables)
    optimizer_disc.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

    # 随机产生噪音数据
    noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)

    with tf.GradientTape() as g:
        fake_images = generator(noise, is_training=True)
        disc_fake = discriminator(fake_images, is_training=True)

        gen_loss = generator_loss(disc_fake)
    # 生成器优化
    gradients_gen = g.gradient(gen_loss, generator.trainable_variables)
    optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    return gen_loss, disc_loss


# 迭代
for step, (batch_x, _) in enumerate(train_data.take(training_steps + 1)):

    if step == 0:
        # S x 100 (s = 128)
        noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)
        # 计算初始损失

        temp1 = generator(noise) # 根据128随机生成的向量，生成128张图片
        temp2 = discriminator( temp1 ) # 对这128张图片，进行判别
        gen_loss = generator_loss( temp2 ) # 查看Loss (相对Real)
        #gen_loss = generator_loss(discriminator(generator(noise)))

        temp1 = discriminator(batch_x) # 对真实的128张照片，进行判别
        temp21 = generator(noise)
        temp22 = discriminator(temp21)  # 对随机生成的128张照片，进行判别
        disc_loss = discriminator_loss(temp1, temp22)
        #disc_loss = discriminator_loss(discriminator(batch_x), discriminator(generator(noise)))
        print("initial: gen_loss: %f, disc_loss: %f" % (gen_loss, disc_loss))
        continue

    # 训练
    gen_loss, disc_loss = run_optimization(batch_x)

    if step % display_step == 0:
        print("step: %i, gen_loss: %f, disc_loss: %f" % (step, gen_loss, disc_loss))

# 结果展示
import matplotlib.pyplot as plt
n = 6
canvas = np.empty((28 * n, 28 * n))
for i in range(n):
    # 还是随机输入
    z = np.random.normal(-1., 1., size=[n, noise_dim]).astype(np.float32)
    # 生成结果
    g = generator(z).numpy()
    # 还原成[0, 1]
    g = (g + 1.) / 2
    g = -1 * (g - 1)
    for j in range(n):
        canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()