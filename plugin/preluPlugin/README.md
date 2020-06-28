# preluPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)


## Description

This plugin applies the Prelu activation with the following equation.

$f(y_i)=\left\{\begin{matrix}
y_i, & if \; y_i >  0 \\ 
a_iy_i, & if \; y_i \leq 0
\end{matrix}\right.$


### Structure

The `preluPlugin` takes one input; `input`.

`input`
input is a tensor with shape `[S, B, E]` where `B` is the batch size.


The `preluPlugin` generates the following output:

`output`
output is a tensor with shape `[S, B, E]` where `B` is the batch size.


## Parameters

`preluPlugin` has plugin creator class `PreluPluginCreator` and plugin class `PreluPlugin`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                               | Description
|----------|-----------------------------------------|-------------------------------------------------------------------
|`int`     |`nbWeights`                              | Number of Weights (supports only 1)
|`bool`    |`channel_shared`                         | When `channelShared = false`, a single scale factor is used for all channels. When `channelShared = true`, scale factors are provided are normalized by the channel size.
|`Weights` |`weights`                                | Slope parameters with [type, parameter memory pointer, number of elements]

## Additional resources

-   [PRELU](https://arxiv.org/abs/1502.01852)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

June 2020
This is the first release of this `README.md` file.


## Known issues

Compatibility is not verified yet.
~~This plugin only supports GPUs with compute capability >= 7.0. For more information see the [CUDA GPU Compute Capability Support Matrix](https://developer.nvidia.com/cuda-gpus#compute)~~
