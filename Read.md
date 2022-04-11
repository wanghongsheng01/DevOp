# OneFlow UserOp 开发
对应视频地址：
```shell
oneflow-13:/tank/meeting_records/20201113-UserOp01-RegisterOp-YaoChi.mp4
oneflow-13:/tank/meeting_records/20201113-UserOp02-RegisterKernel-YaoChi.mp4
oneflow-13:/tank/meeting_records/20201113-UserOp03-TestCase-YaoChi.mp4 
```

如何学习 PyTorch 算子源码的文章及视频：https://github.com/Oneflow-Inc/OneTeam/issues/147

## 技术组成

- C++
- CUDA
- gdb

## 开发流程

OneFlow 其实可以看作是python的一个插件（`_oneflow_internal.cpython-38-x86_64-linux-gnu.so`）。UserOp 其实最终可以生成so。大家查看https://docs.oneflow.org/extended_topics/user_op.html 相关的代码可以知道。

但是在OneFlow框架开发中，其实比我们的教程还要简单，我们不需要像教程中那样自己写Makefile，这些工作都有OneFlow自身的编译系统解决了。

我们要做的，只是将代码放置在对应的目录(`oneflow/user/`)即可。



### op 与 kernel

因为 OneFlow 是为分布式设计的系统，所以它的 Op 和 Kernel 的概念比较独特。

简单而言，Op 是关系描述逻辑上的概念，Kernel 关心具体物理设备上的运算。



### op 注册

基本概念：

- `op_type_name`：每个op都有一个唯一的ID，就是 `op_type_name`，它其实就是一个全局唯一的字符串。注册OP时，开发者自己选择`op_type_name`。

我们通过宏 `REGISTER_USER_OP` 来注册一个 Op。因为 Op 只是关心逻辑的概念，所以等下可以看到，注册的过程中，也只是在关心逻辑上的推导（而不是具体的计算）

这个宏实际返回一个 `OpRegistry` 对象，我们通过这个对象里的方法，来设置 Op 的属性。常见的需要设置的有：

- 输入，通过 Input 等方法设置
- 输出，通过 Output 等方法设置
- 设置属性，有些信息，并不属于 Blob，但是它又在执行 op/kernel 的过程被需要。那么可以以属性的方式设置。 `ctx->Attr<bool>("stride_partition")`

以上的输入、输出、属性，他们都是使用字符串作为名字（ID）。这样，我们在后续的各种回调函数中，可以通过他们的名字，获取到对应的对象。



注册重要的回调函数：

- `SetTensorDescInferFn`：用于根据op的输入blob，推导op的输出张量
- `SetGetSbpFn`：用于设置当前 Op 的 SBP Signature
- `SetInputArgModifyFn`/`SetOutputArgModifyFn`： 用于设置输入/输出 Blob 的特殊属性
- `SetBatchAxisInferFn` ：用于设置 batch axis



以形状推导为例：

```cpp
  user_op::TensorDesc* a = ctx->TensorDesc4ArgNameAndIndex("a", 0);
  user_op::TensorDesc* b = ctx->TensorDesc4ArgNameAndIndex("b", 0);
```

可以通过 `XXX4NameAndIndex` 方法，根据 字符串ID，拿到对应的Desc对象，因为此时没有数据，只关心形状、数据类型。

```cpp
*out = *a;
  int64_t m, n, k;  // tensor a (no trans): m*k, tensor b (no trans): k*n
  if (!transpose_a) {
    m = a->shape().At(num_axes - 2);
    k = a->shape().At(num_axes - 1);
  } else {
    m = a->shape().At(num_axes - 1);
    k = a->shape().At(num_axes - 2);
  }
  if (!transpose_b) {
    CHECK_EQ_OR_RETURN(k, b->shape().At(num_axes - 2));
    n = b->shape().At(num_axes - 1);
  } else {
    CHECK_EQ_OR_RETURN(k, b->shape().At(num_axes - 1));
    n = b->shape().At(num_axes - 2);
  }
  out->mut_shape()->Set(num_axes - 2, m);
  out->mut_shape()->Set(num_axes - 1, n);
```



接着，看 `SetGetSbpFn` 的细节，建议先阅读 [这篇](https://docs.oneflow.org/basics_topics/essentials_of_oneflow.html#sbp)和(这篇)[https://docs.oneflow.org/arch_design/infer_sbp_signature.html]后，再参阅其它已有的实现。

`SetInputArgModifyFn` ，它使得我们有机会去修改输入的张量的一些性质。关于输入、输出张量有哪些性质可以修改，可以参考 `oneflow/core/operator/arg_modifier_signature.proto`。

```
message InputBlobModifier {
  optional bool is_mutable = 1 [default = false];
  optional bool use_header_only = 2 [default = false];
  optional bool requires_grad = 3 [default = false];
}
```

- is_mutable：是否可修改
- use_header_only：某些情况下，我们并不需要张量的数据，比如`scatter_like`
- requires_grad：如果某些输入张量，不需要反向梯度，则设置为false



### kernel 注册与开发

类似的，Kernel是通过宏完成的注册 `REGISTER_USER_KERNEL`，实际操作的`OpKernelRegistry`的方法。

因为我们知道，Kernel 它是实际进行计算的功能单元，所以它比较关心的：

- 用什么物理设备进行计算
- 针对怎样的数据类型进行计算
- 计算的代码写在哪里



```
  REGISTER_USER_KERNEL("dim_gather")                                                     \
      .SetCreateFn<DimGatherKernel<DeviceType::kCPU, float, int32_t>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kCPU)                               \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::float) \
                       & (user_op::HobDataType("index", 0) == GetDataType<itype>::int32_t));
```

如以上：

- SetIsMatchedHob，它接收表达式，表达式用于进行设备和数据类型的判断。当结果为True时，代表与当前Kernel匹配。以上的代码中，就确认了，只有在CPU设备上、input类型为float、index类型为int32时，当前Kernel才与之匹配
- SetCreateFn，上文提到的所谓”当前的Kernel“，其实就是对应的计算逻辑。这些计算逻辑，都被封装在类中。我们通过 `SetCreateFn`，来告诉OneFlow框架，匹配的是那个类



所以，以上代码的完整概念就是：”在CPU设备上、input类型为float、index类型为int32时“，使用 `DimGatherKernel<DeviceType::kCPU, float, int32_t>`进行计算。



### Kernel 类的实现

```
template <DeviceType device_type, typename T>
class ReluKernel final : public user_op::OpKernel {
public:
  ReluKernel() = default;
  ~ReluKernel() = default;

private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    MyRelu<T>(ctx->device_ctx(),
           in_tensor->shape().elem_cnt(),
           in_tensor->dptr<T>(), 
           out_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
```

User Op 要求，用户自定义的Kernel 类，必须继承自 `user_op::OpKernel`，并且通过重写其中的 `Compute` 等方法，将业务逻辑，填充到其中。

- Compute：它有两种重载，一种只有一个参数ctx；另外一种多了个两个参数，如果除了ctx提供之外的信息，我们还想在各个Compute调用之间交互信息的话，我们就可以使用有三个参数的Compute。具体可以参考 https://github.com/Oneflow-Inc/OneTeam/issues/119#issuecomment-994215271 。

```
  virtual void Compute(KernelComputeContext* ctx, OpKernelState*, const OpKernelCache*) const { Compute(ctx); }
  virtual void Compute(KernelComputeContext*) const { LOG(INFO) << "UNIMPLEMENTED"; }
```

- AlwaysComputeWhenAllOutputsEmpty：一般返回false



### CUDA 编程基本概念

CUDA入门常识：CUDA编程的代码，核心运行在GPU上，与我们普通编程运行的代码在CPU上，这个是最大的不同。

因为有物理设备的不同，所以涉及到了”切换“，因为有”切换“的需求，所以CUDA编程，扩展了C++的语法。

```
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

以上，这句话是进行”切换“：

```cpp
MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
```

其次，不是所有的函数，都可以用来做切换，而只有”核函数“才可以。CUDA发明了新的关键字 `__global__`，被它修饰的函数，是核函数。

除此之外，CUDA的扩展语法 `__host__`，`__device__`，他们也与CPU、GPU的独立隔离有关。

OneFlow 中提供了对应的宏，来启动核函数：

```
RUN_CUDA_KERNEL(MatAdd, ctx->device(), numBlocks, A, B, C)
```



### Python Wrapper

我们知道，OneFlow 实际代码都在C++层运行，但是给用户的接口的Python接口。因此我们需要在Python层进行封装，供用户使用，核心是调用`flow.user_op_builder`：

```python
@oneflow_export("gather")
def gather(
    params: remote_blob_util.BlobDef,
    indices: remote_blob_util.BlobDef,
    validate_indices: Optional[remote_blob_util.BlobDef] = None,
    axis: Optional[int] = None,
    batch_dims: int = 0,
    name: Optional[str] = None,
) -> remote_blob_util.BlobDef:
        #...
        myop = (
            flow.user_op_builder(
                name if name is not None else id_util.UniqueStr("Gather_")
            )
            .Op("gather") # op_type_name
            .Input("in", [params]) # 设置输入，对用了C++ op设置的输入
            .Input("indices", [indices])
            .Output("out") # 设置输出，对应了C++ op设置的输出
            .Attr("axis", int(axis)) # 设置属性，对应了C++ Attr设置的属性
            .Build() # build，完成了封装
            
        out = myop.InferAndTryRun()
           	     .RemoteBlobList()[0] # 和 SoleOutputBlob() 等价
        return out
```





### 测试案例

测试案例，可以参考 https://github.com/Oneflow-Inc/OneTeam/issues/82

坑：

- 对于 GPU 的测试，需要设置修饰符，否则过不了CPU-ONLY的测试

```
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dim_gather_float_gpu(test_case):
        arg_dict = _gen_arg_dict("gpu", "float", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_dim_gather_with_samples(test_case, *arg)
```

- 为了应对 CPU-ONLY，在C++层次的代码中，凡是涉及 CUDA的编程的，应该使用WITH_CUDA条件编译，使得它在CPU版本中不存在。

- 如果要多卡测试diff，watch_diff 是不能直接获取多卡（逻辑上的）diff的，需要先在单机上获取，再利用boxing机制转换。

  ```
          def do_where(condition, x, y):
              with flow.scope.placement(device_type, "0:0"):
                  x_var = flow.get_variable(
                      "x",
                      shape=x.shape,
                      dtype=flow.float,
                      initializer=flow.constant_initializer(0),
                  )
                  x_var = flow.cast_to_current_logical_view(x_var)
                  x_var = x_var + x
                  y_var = flow.get_variable(
                      "y",
                      shape=y.shape,
                      dtype=flow.float,
                      initializer=flow.constant_initializer(0),
                  )
                  y_var = flow.cast_to_current_logical_view(y_var)
                  y_var = y_var + y
  
              z = flow.where(condition, x_var, y_var)
  
              with flow.scope.placement(device_type, "0:0"):
                  flow.optimizer.SGD(
                      flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
                  ).minimize(z)
  
              flow.watch_diff(x_var, dz_dx_watcher)
              flow.watch_diff(y_var, dz_dy_watcher)
              return z
  ```

  
### 本地 EAGER 模式测试
目前需要自己完成本地的 EAGER 测试，设置环境变量 `ONEFLOW_TEST_ENABLE_EAGER` 为1，再运行测试脚本即可，如：
单机测试：
```shell
ONEFLOW_TEST_ENABLE_EAGER=1 python test_relu.py --verbose --failfast
```
单机两卡测试：
```shell
ONEFLOW_TEST_ENABLE_EAGER=1 ONEFLOW_TEST_DEVICE_NUM=2 python3 test_relu.py --verbose --failfast
```


## 常见宏与方法
### NdIndexOffsetHelper

用于一维坐标和高维坐标转换的工具类




### OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE

用于把宏做笛卡尔积展开。比如，我们有一个宏：

```
#define myadd(x, y)
myadd(x0, y0)
myadd(x1, y1)
myadd(x0, y1)
myadd(x1, y0)
```

使用 `OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE`会很简洁：

```
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(myadd, (x0 x1), (y0 y1))
```

虽然之前这个宏被大量用于kernel注册，但是现在并不推荐。



### RUN_CUDA_KERNEL
```
#define RUN_CUDA_KERNEL(func, device_ctx_ptr, thread_num, ...)     \
  func<<<SMBlocksNum4ThreadsNum(thread_num), kCudaThreadsNumPerBlock, 0,
         (device_ctx_ptr)->cuda_stream()>>>(__VA_ARGS__)

```


### XPU_1D_KERNEL_LOOP
通用的CPU、GPU宏


## 文件组织与 PR 提交
### 文件组织方式
OneFlow Op 开发关于 Kernel 代码实现的文件组织方式，常见有 方案A 和 方案B 两种。选择原则是，尽量选择方案A，少数情况下可以选择方案B，解释如下。

#### 方案 A
方案A，将具体的计算独立出，放在 Util Class 中，这样做的好处是这个算子的计算过程可以被其他算子复用，比如 FooKenrelUtil::Forward 可以被 BarKernel::Compute 调用。

文件组织及其代码分工如下：
- `*_kernels.cpp`：kernel 类、Kernel 的注册
- `*_kernel_util.h`：Kernel Util 类的声明
- `*_kernel_util.cpp`： Kernel Util 类的 CPU 特例化实现
- `*_kernel_util.cu`： Kernel Util 类的 GPU 特例化实现

这样做的坏处是，  Kernel Util 类的 CPU 特例化实现 和  Kernel Util 类的 GPU 特例化实现 必须要 **显式实例化**，否则会出现链接错误 `undefined reference`。

有时候 KernelUtil 的模版参数会比较复杂，使显式实例化变得困难，甚至有可能复杂到无法显式实例化，这个时候我们就得采用方案 B。

#### 方案 B
在 B 方案中，Kernel Util 的特例化实现和 Kernel 注册，放在了同一个文件中，使得在注册 Kernel 的过程中，就隐式地完成了 Kernel Class 及 Kernel Util Class 的模板实例化。

文件组织及其代码分工如下：
- `*_kernel.h`：Kernel 类的声明和定义、Kernel Util 的声明，用于 Kernel 注册的宏的定义
- `*_kernel.cpp`：Kernel Util 的 CPU 特例化实现，注册 CPU Kernel
- `*_kernel.cu`：Kernel Util 的 GPU 特例化实现，注册 GPU Kernel

这样做的坏处是，一旦用户想在 BarKernel::Compute 的实现中复用 FooKernelUtil 的逻辑，他就得 #include “foo_kernel.h”，同时会把 FooKernel 的逻辑也暴露出去，这破坏了隐藏原则。

### PR 提交模板
OP PR 的模板在源码仓库的 [.github/PULL_REQUEST_TEMPLATE](https://github.com/Oneflow-Inc/oneflow/blob/master/.github/PULL_REQUEST_TEMPLATE/op_template.md) 路径下，OP PR 的提交应该按照模板填写对应信息。

也可以通过 URL 快速创建PR，将以下地址中的 `【branch_name】` 替换为对应的分支名即可：

https://github.com/Oneflow-Inc/oneflow/compare/master...【branch_name】?template=op_template.md

