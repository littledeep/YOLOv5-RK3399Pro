from rknn.api import RKNN
if __name__ == '__main__':
    # Create RKNN object
    rknn = RKNN()


# Load rknn model
ret = rknn.load_rknn('./best_as_200.rknn')
if ret != 0:
    print('Load RKNN model failed.')
    exit(ret)
# init runtime
ret = rknn.init_runtime(target='rk3399pro', rknn2precompile=True)
if ret != 0:
    print('Init runtime failed.')
    exit(ret)
# Note: the rknn2precompile must be set True when call init_runtime
ret = rknn.export_rknn_precompile_model('./best_pre_compile.rknn')
if ret != 0:
    print('export pre-compile model failed.')
    exit(ret)
rknn.release()