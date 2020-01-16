---
layout: post
title: Binary neural networks in Gradient
categories: gradient whats-new binary bitwise
excerpt_separator: <!--more-->
---

As you might have heard, today Apple has [acquired our Seattle neighbor Xnor.ai](https://www.businessinsider.com/apple-reportedly-buys-xnor-ai-200-million-2020-1) for $200M.
The company's main product is a mechanism to run neural networks on low-power devices,
and its core is just [50 lines of code](https://github.com/allenai/XNOR-Net/blob/master/models/alexnetxnor.lua).
It achieves then efficiency by performing operations _en masse_ on individual bits instead of the normal
32- and recently 16-bit floating point numbers.

For the last few months we have been working to bring bitwise operations to Gradient,
and yesterday we finally got the first relatively stable build based on the latest TensorFlow
from 1.x family: 1.15 (previous versions of Gradient up to Preview 6.4 were based off TensorFlow 1.10).
The new version brings support for many new features, among them `tf.bitwise` and `gen_bitwise_ops` modules.
In the light of the Xnor.ai acquisition news I decided to publish these bits with a simple sample code
to a [work-in-progress branch](https://github.com/losttech/Gradient-Samples/tree/updates/tf1.15),
so you could start trying them early. You can view
[the new sample code for bitwise ops](https://github.com/losttech/Gradient-Samples/commit/2b0116d21a770e8bd7497ef3d759b3703925074e)
in [Gradient-Samples](https://github.com/losttech/Gradient-Samples) repository, but it is as easy as
```csharp
Tensor xor = tf.bitwise.bitwise_xor(x, y);
Tensor bitcount = gen_bitwise_ops.population_count_dyn(xor);
```

Stay tuned for the official release with TensorFlow 1.15 support. It is coming soon!
