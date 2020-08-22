---
layout: post
title: TensorFlow for .NET Release Candidate
categories: gradient tensorflow whats-new
excerpt_separator: <!--more-->
---

Today **LostTech.TensorFlow**, our C# binding to TensorFlow, gets its first Release
Candidate available to public. It brings a stable API based on TensorFlow 1.15, that includes
tensors, graphs, and sessions, along with Keras and an optional
[eager execution mode](https://www.tensorflow.org/guide/eager), and much more.

Check out [the product webpage](https://losttech.software/gradient.html) for the
[overview of available features](https://losttech.software/gradient.html).
See [our samples](https://github.com/losttech/Gradient-Samples/) or just
install the [NuGet package](https://nuget.org/packages/LostTech.TensorFlow) and start your journey
into deep learning with .NET!

This Release Candidate comes with a go-live license, meaning unlike the previews, it has no
expiration date.

This release is completely free for individuals and small businesses, while
larger enterprises can get a pay-as-you-go license through their Azure Subscription, which includes
some trial credits. Please, contact [Sales](https://losttech.software/buy_gradient.html) if you
need an offline license.

<!--more-->

<h3>Contents</h3>
* TOC
{:toc}

## Why LostTech.TensorFlow

### Full power

To this date [people _struggle_ to implement](https://github.com/SciSharp/TensorFlow.NET/issues/352)
state of the art algorithms for **computer vision (YOLOv3)** and
**language processing (GPT and BERT)** using **TensorFlow.NET** due to the
*lack of required features*.

In comparison, our [**GPT-2 demo**](https://habr.com/post/453232/) was **made in one week** just
a month after Open AI released the smaller model back in February 2019, thanks to the API parity
with TensorFlow for Python.

![TensorBoard screenshot](https://www.tensorflow.org/images/mnist_tensorboard.png)

### Best performance

We are using optimized builds of TensorFlow provided by Google, which come with all
performance-related bells and whistles.

We are leaving the full comparison to reviewers, but in
[a simple test](https://github.com/losttech/Gradient-Perf/) we made out of **TensorFlow.NET**'s
official sample, **LostTech.TensorFlow** came about **18% faster** to train a convolutional neural
network.

The use of official **TensorFlow** builds with **LostTech.TensorFlow** enables you to take advantage
not only of **NVidia GPUs**, but also of **AMD accelerators** using
[tensorflow-rocm](https://pypi.org/project/tensorflow-rocm/1.15.4/),
and **Tensor Processing Units** in Google Cloud.

![18% faster than TF.NET](/images/perf-vs-tf.net.png)

### Integration with Unity ML Agents

**LostTech.TensorFlow** includes support for interacting with virtual environments
built using [**Unity ML Agents**](https://github.com/Unity-Technologies/ml-agents).

This allows to train deep learning models using .NET in detailed simulated environments,
that can be transferred to power real-life robots. This also opens an opportunity to expand
in-game AI characters with human-like features.

![Trained bot playing 3DBall](/images/3DBall-Trained.gif)

### Commercial support

We provide full support for **LostTech.TensorFlow** on AMD64/x64 hardware on Linux, MacOS,
and Windows. We will resolve any issues caused by the product in accessing supported TensorFlow 1.15
APIs.

## What's new in this release

