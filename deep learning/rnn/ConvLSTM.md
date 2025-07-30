在传统lstm中,所有的w都是全连接层,在引入conv后$ x_i*w $和$ h_{i-1}*w $变成了$ Conv2d(x_i) $和$ Conv2d(h_{i-1}) $,更加适合视频等时序数据

