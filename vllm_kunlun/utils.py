#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
# This file is a part of the vllm-kunlun project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys
import vllm

from torch.utils._python_dispatch import TorchDispatchMode
import vllm_kunlun.platforms.envs as xenvs
from vllm.utils import weak_ref_tensor
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
    get_origin,
    get_args,
    List,
)
import torch
from torch.library import Library
import inspect
import typing


def redirect_output():
    """
    Redirect output to a specified directory and name the log files as pp=0_rank=X or pp=1_rank=X.
    If it is the first process of the first process group, use pp=0; otherwise, use pp=1.

    Args:
        No parameters.

    Returns:
        No return value, directly modify the file descriptors of sys.stdout and sys.stderr.
    """
    from vllm.distributed import get_tensor_model_parallel_rank, get_pp_group

    rank = get_tensor_model_parallel_rank()
    dir_path = xenvs.VLLM_MULTI_LOGPATH
    os.makedirs(dir_path, exist_ok=True)
    if get_pp_group().is_first_rank:
        log_file = os.path.join(dir_path, f"pp=0_rank={rank}.log")
    else:
        log_file = os.path.join(dir_path, f"pp=1_rank={rank}.log")
    fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())
    os.close(fd)


def multi_log_monkey_patch(func):
    """
    Monkey patch function for logging multiple times, used to test log redirection functionality.
    This function will print a log message each time the patched function is called.

    Args:
        func (function): The original function to be patched.

    Returns:
        function: A wrapped new function that prints a log message each time it is called.
    """

    def wrapper(*args, **kwargs):
        print("[monkey patch] ensure_model_parallel_initialized")
        func(*args, **kwargs)
        redirect_output()

    return wrapper


if xenvs.ENABLE_VLLM_MULTI_LOG:
    print("ENABLE_VLLM_MULTI_LOG monkey--------")
    vllm.distributed.ensure_model_parallel_initialized = multi_log_monkey_patch(
        vllm.distributed.ensure_model_parallel_initialized
    )


class StageHookPre(object):
    def __call__(self, *args, **kwargs):
        """
            This method will be automatically executed when the object is called.
        If the current attention metadata is not None and a token has been processed, print "Per Token Start"; otherwise, print "First Token Start".

        Args:
            args (tuple, optional): Variable length argument list, default is an empty tuple.
            kwargs (dict, optional): Keyword arguments, default is an empty dictionary.

        Returns:
            None: No return value.
        """
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None:
            if attn_metadata.num_decode_tokens == 0:
                print("First Token Start", flush=True)
            else:
                print("Per Token Start", flush=True)


class StageHookPost(object):
    def __call__(self, *args, **kwargs):
        """
            If the current context's attention metadata is not None and num_decode_tokens equals 0, print "First Token End".
        Otherwise, print "Per Token End".

        Args:
            args (Tuple[Any]): Variable length argument list, unused parameters are passed in.
            kwargs (Dict[str, Any]): Keyword arguments, unused parameters are passed in.

        Returns:
            None: No return value.
        """
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is not None:
            if attn_metadata.num_decode_tokens == 0:
                print("First Token End", flush=True)
            else:
                print("Per Token End", flush=True)


class ModuleLoggingHookPre(object):
    def __init__(self):
        """
            Initialization function to initialize the indentation list and name list.
        The indentation list is used to store the indentation information of each line,
        and the name list is used to store the name of each variable or function.
        """
        self.indent_list = list()
        self.indent_list.append("")
        self.name_list = list()

    def __call__(self, *args, **kwargs):
        """
            This method overrides the __call__ method and is used when the class is instantiated.
        It increases the current indentation by one Tab and records the current class name.
        It prints the start information, flush=True means it will be output to the console immediately.

        Args:
            args (tuple): Variable length argument list, default is an empty tuple.
            kwargs (dict): Keyword arguments, default is an empty dictionary.

        Returns:
            None.
        """
        self.indent_list.append(self.indent_list[-1] + "\t")
        self.name_list.append(args[0].__class__.__module__ + args[0].__class__.__name__)
        print(self.indent_list[-1] + self.name_list[-1] + " Start", flush=True)


class ModuleLoggingHookPost(object):
    def __init__(self, indent_list, name_list):
        """
            Initialization function to set the indentation list and name list.

        Args:
            indent_list (List[str]): A list of indentation strings for each node, indexed from 0.
            name_list (List[str]): A list of name strings for each node, indexed from 0.
            Note: The indentation list and name list should have the same length, otherwise it will cause an error.

        Returns:
            None: No return value, directly modifies the instance's attributes.
        """
        self.indent_list = indent_list
        self.name_list = name_list

    def __call__(self, *args, **kwargs):
        """
            This method is called when the object is invoked.
        Args:
            *args, **kwargs: Variable length argument list and keyword argument dictionary, unused.
        Returns:
            None: No return value.
        """
        print(self.indent_list[-1] + self.name_list[-1] + " Module End", flush=True)
        self.indent_list.pop()
        self.name_list.pop()


if xenvs.ENABLE_VLLM_MODULE_HOOK:
    from torch.nn.modules.module import (
        register_module_forward_pre_hook,
        register_module_forward_hook,
    )

    module_logging_hook_pre = ModuleLoggingHookPre()
    module_logging_hook_post = ModuleLoggingHookPost(
        module_logging_hook_pre.indent_list, module_logging_hook_pre.name_list
    )
    register_module_forward_pre_hook(module_logging_hook_pre)
    register_module_forward_hook(module_logging_hook_post)
else:
    module_logging_hook_pre = None
    module_logging_hook_post = None


class LoggingDispatchMode(TorchDispatchMode):
    def __init__(self):
        """
            Initialization function to initialize the attributes and methods of the class.
        Some initialization operations can be performed here, such as setting default values.
        """
        super().__init__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        Override the default dispatch behavior of torch.nn.Module.
            This function will be called before and after each method call on this module.
            It can be used to log information about the method calls.

            Args:
                func (function): The function that is being called on this module.
                types (Tuple[str]): A tuple of strings representing the type signatures of the arguments.
                    See torch.types for more details.
                args (Tuple[Any], optional): The positional arguments passed to the function. Defaults to ().
                kwargs (Dict[str, Any], optional): The keyword arguments passed to the function. Defaults to {}.

            Returns:
                Any: The result returned by the function.
        """
        global module_logging_hook_pre
        if module_logging_hook_pre is not None:
            indent = module_logging_hook_pre.indent_list[-1]
        else:
            indent = "\t"
        print(indent + "{} calling".format(func), flush=True)
        result = func(*args, **(kwargs or {}))
        print(indent + "{} called".format(func), flush=True)

        return result


class CUDAGraphInnerWatcher(TorchDispatchMode):

    def __init__(self, name_list):
        """
            Initialization function to save the name list to the class attribute.
        It also creates a dictionary to keep track of the tensors that have been traced.

        Args:
            name_list (List[str]): A list of names of tensors to be tracked.

        Returns:
            None.
        """
        self.name_list = name_list
        self.traced_tensor = dict()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        Override the default dispatch behavior of PyTorch tensors to track
        the tracing process. If the result of a function call is a tensor on CUDA,
        it will be added to the traced_tensor dictionary with the name of the function.

        Args:
            func (Callable): The function to be called.
            types (Tuple[Type]): The type hints of the function.
            args (Tuple[Any], optional): Positional arguments for the function. Defaults to ().
            kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for the function. Defaults to None.

        Returns:
            Any: The result of the function call.
        """
        result = func(*args, **(kwargs or {}))
        if isinstance(result, torch.Tensor) and result.is_cuda:
            if func._name in self.name_list:
                self.traced_tensor[func._name] = weak_ref_tensor(result)
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
            Clear the traced_tensor and name_list, and call the parent class's __exit__ method.

        Args:
            exc_type (Optional[Type[BaseException]]): The type of the exception, default is None.
            exc_val (Optional[BaseException]): The value of the exception, default is None.
            exc_tb (Optional[TracebackType]): he traceback object, default is None.

        Returns:
            None.
        """
        for name, value in self.traced_tensor.items():
            print(name, value)
        self.traced_tensor.clear()
        self.name_list.clear()
        super(CUDAGraphInnerWatcher, self).__exit__(exc_type, exc_val, exc_tb)


def patch_annotations_for_schema(func):
    """
    At runtime, replace list[int] and Optional[list[int]] in the function signature with typing.List[int] and Optional[typing.List[int]]
    so that torch.library.infer_schema can recognize it.
    """
    sig = inspect.signature(func)
    new_params = []

    for name, param in sig.parameters.items():
        ann = param.annotation

        # If it is Optional[T]
        if get_origin(ann) is typing.Union and type(None) in get_args(ann):
            inner_type = [a for a in get_args(ann) if a is not type(None)][0]
            if get_origin(inner_type) is list:  # Optional[list[int]]
                inner_args = get_args(inner_type)
                new_ann = Optional[List[inner_args[0] if inner_args else typing.Any]]
                param = param.replace(annotation=new_ann)

        # If it is a direct list[int]
        elif get_origin(ann) is list:
            args = get_args(ann)
            new_ann = List[args[0] if args else typing.Any]
            param = param.replace(annotation=new_ann)

        new_params.append(param)

    func.__signature__ = sig.replace(parameters=new_params)
    return func


def supports_custom_op() -> bool:
    """supports_custom_op"""
    return hasattr(torch.library, "custom_op")


vllm_lib = Library("vllm", "FRAGMENT")  # noqa


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
    dispatch_key: str = "CUDA",
    tags: tuple[torch.Tag, ...] = (),
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.
    """
    if not supports_custom_op():
        from vllm.platforms import current_platform

        assert not current_platform.is_cuda_alike(), (
            "cuda platform needs torch>=2.4 to support custom op, "
            "chances are you are using an old version of pytorch "
            "or a custom build of pytorch. It is recommended to "
            "use vLLM in a fresh new environment and let it install "
            "the required dependencies."
        )
        return

    import torch.library

    if hasattr(torch.library, "infer_schema"):
        patched_func = patch_annotations_for_schema(op_func)
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
