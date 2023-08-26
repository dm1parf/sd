def convertModuleNames(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }
