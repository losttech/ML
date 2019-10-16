---
layout: post
title: What's New in Gradient Preview 6.4?
categories: gradient whats-new
excerpt_separator: <!--more-->
---

We released Gradient Preview 6.4 on Oct 15, 2019. It brings several new features and bug fixes:

- feature: inheriting from TensorFlow classes enables defining custom Keras layers and models
- feature: improved automatic conversion .NET types to TensorFlow
- feature: fast marshalling from .NET arrays to NumPy arrays
- bug fix: it is now possible to modify collections belonging to TensorFlow objects
- bug fix: enumerating TensorFlow collections could crash in multithreaded environment
- new samples: [ResNetBlock](https://github.com/losttech/Gradient-Samples/tree/master/ResNetBlock
) and [C# or Not](https://lostmsu.github.io/Not-CSharp/)
- train models in [Jupyter F# notebook in your browser](https://notebooks.azure.com/lost/projects/gradient-samples/html/FashionMNIST.ipynb)
hosted for free by Microsoft Azure
- preview expiration: extended to March 2020

[![F# Notebook Screenshot](/images/NotebookScreenshot.png)](https://notebooks.azure.com/lost/projects/gradient-samples/html/FashionMNIST.ipynb)

<!--more-->

<h3>Contents</h3>
* TOC
{:toc}

## Inheriting from TensorFlow classes

With Gradient Preview 6.4 it is now possible to inherit from most classes in `tensorflow` namespace.
This enables defining custom Keras models and layers as suggested in the official
[TensorFlow tutorial](https://www.tensorflow.org/tutorials/customization/custom_layers).

**NOTE:** In Preview 6.4 defining custom `Model` class requires every layer to be explicitly tracked
using a call to `Model.Track` function, so that TensorFlow could keep `.layers` collection in sync.
Failure to do so may cause training to fail at runtime. See it in the [ResNet sample](
    https://github.com/losttech/Gradient-Samples/blob/f820b920eb53e6f84b93c74cdade5e60e30af392/ResNetBlock/ResNetBlock.cs#L22
).

Here is an example of a custom composable model: `ResNetBlock`, that has 3 convolutional layers,
joined using batch normalization, and a "skip-connection" node, that connects input
directly to the output.

**NOTE:** `BatchNormalization` layer requires TensorFlow 1.14 to work properly. In 1.10 it is unstable.

```csharp
public class ResNetBlock: Model {
    const int PartCount = 3;
    readonly PythonList<Conv2D> convs = new PythonList<Conv2D>();
    readonly PythonList<BatchNormalization> batchNorms = new PythonList<BatchNormalization>();
    readonly PythonFunctionContainer activation;
    readonly int outputChannels;

    public ResNetBlock(int kernelSize, int[] filters,
                       PythonFunctionContainer activation = null)
    {
        this.activation = activation ?? tf.keras.activations.relu_fn;
        for (int part = 0; part < PartCount; part++) {
            this.convs.Add(this.Track(part == 1
                ? Conv2D.NewDyn(
                    filters: filters[part],
                    kernel_size: kernelSize,
                    padding: "same")
                : Conv2D.NewDyn(filters[part], kernel_size: (1, 1))));
            this.batchNorms.Add(this.Track(new BatchNormalization()));
        }

        this.outputChannels = filters[PartCount - 1];
    }

    public override dynamic call(
            object inputs,
            ImplicitContainer<IGraphNodeBase> training = null,
            IEnumerable<IGraphNodeBase> mask = null)
    {
        return this.CallImpl((Tensor)inputs, training?.Value);
    }

    object CallImpl(IGraphNodeBase inputs, dynamic training) {
        IGraphNodeBase result = inputs;

        var batchNormExtraArgs = new PythonDict<string, object>();
        if (training != null)
            batchNormExtraArgs["training"] = training;

        for (int part = 0; part < PartCount; part++) {
            result = this.convs[part].apply(result);
            result = this.batchNorms[part].apply(result, kwargs: batchNormExtraArgs);
            if (part + 1 != PartCount)
                result = ((dynamic)this.activation)(result);
        }

        result = (Tensor)result + inputs;

        return ((dynamic)this.activation)(result);
    }

    public override dynamic compute_output_shape(TensorShape input_shape) {
        if (input_shape.ndims == 4) {
            var outputShape = input_shape.as_list();
            outputShape[3] = this.outputChannels;
            return new TensorShape(outputShape);
        }

        return input_shape;
    }

    ...
}
```

Functions like `Model.call`, overridden in the above sample have many overloads.
You have to override only the ones, that TensorFlow will actually call. If you failed to override
the proper one, you will get either `AttributeError`, or `TypeError` with message
"_No method matches given arguments_" at runtime, telling you which one is missing. This happens
before training begins, so remember to test your model on a small data sample
if you do heavy preprocessing. You might also need to define new overloads.

## Marshalling .NET collections to NumPy arrays

Prior to Preview 6.4 creating a NumPy array from .NET array would copy elements
one by one, and a custom conversion op would be performed for each one
consuming both lots of time and memory.

In 6.4 we introduced an extension method `.NumPyCopy()` for arrays,
`IEnumerable<T>`, and `ReadOnlySpan<T>`, that copies data to TensorFlow accessible memory
very quickly. Unfortunately, we still do not support arrays over 2GiB because .NET runtime
does not support them. Please vote for large
 [Spans](https://github.com/dotnet/coreclr/issues/27040)
 and [arrays](https://github.com/dotnet/coreclr/issues/23132) support in .NET.

Example (from Not C# sample):

```csharp
static ndarray<float> GreyscaleImageBytesToNumPy(byte[] inputs, int imageCount, int width, int height)
    => (dynamic)inputs.Select(b => (float)b).ToArray().NumPyCopy()
        .reshape(new[] { imageCount, height, width, 1 }) / 255.0f;
```

## ResNetBlock sample

This is a simple sample, demonstrating inheritance feature of Preview 6.4. It uses
a simplified ResNet to solve FashionMNIST with fewer parameters.

[Source code](https://github.com/losttech/Gradient-Samples/tree/master/ResNetBlock)
can be found in our [samples repository on GitHub](https://github.com/losttech/Gradient-Samples).

## Not C# sample

This is an advanced sample, that demonstrates data preparation in C#,
training a deep convolutional network, and consuming it from a cross-platform C# application.

The network detects programming language from a code fragment.

[![Not C# app screenshot](/images/NotCSharp-Parts.png)](https://lostmsu.github.io/Not-CSharp/)

Full description of the sample is in a separate [blog post](https://lostmsu.github.io/Not-CSharp/).

## Azure Notebook sample with F\#

This is a port of the FashionMNIST sample to [Azure Notebooks](https://notebooks.azure.com/), which
allows training and running models for free in a Jupyter environment provided by Microsoft Azure.

[View the notebook](https://notebooks.azure.com/lost/projects/gradient-samples/html/FashionMNIST.ipynb)
in the browser. (**Clone** if you wish to edit it;
it will also enable syntax highlighting and autocomplete)

[![F# Notebook Screenshot](/images/NotebookScreenshot.png)](https://notebooks.azure.com/lost/projects/gradient-samples/html/FashionMNIST.ipynb)

## Conclusion

Try Gradient Preview 6.4 from [NuGet](https://www.nuget.org/packages/Gradient/)
or [directly in your browser with F#](https://notebooks.azure.com/lost/projects/gradient-samples/html/FashionMNIST.ipynb).
