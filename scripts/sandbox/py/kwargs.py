def print_args(arg1, arg2, arg3):
    print("Arg1:",arg1)
    print("Arg2:",arg2)
    print("Arg3:",arg3)

print_args(1,2,3)
args = [2,3]
print_args(1,*args)
kwargs = {"arg3":3, "arg2":2}
print_args(1,**kwargs)

