# 型チェック用のデコレータ
import inspect
import functools
def args_type_check(func):
    @functools.wraps(func)
    def args_type_check_wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        for arg_key, arg_val in sig.bind(*args, **kwargs).arguments.items():
            annot = sig.parameters[arg_key].annotation
            request_type = annot if type(annot) is type else inspect._empty
            if request_type is not inspect._empty and type(arg_val) is not request_type:
                error_msg = '引数"{}"の型が対応していません．（対応している型：{}，指定された型：{}）'
                raise TypeError(error_msg.format(arg_key, request_type, type(arg_val)))
        return func(*args, **kwargs)
    return args_type_check_wrapper
