import torch
import os

from engine import evaluate

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def post_training_static_quantization(args, model, data_loader_train, data_loader_test):

    model.to('cpu')
    model.eval()

    # Fuse Conv, bn and relu
    #model.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    model.qconfig = torch.ao.quantization.default_qconfig
    print(model.qconfig)
    torch.ao.quantization.prepare(model, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    #print('\n Inverted Residual Block:After observer insertion \n\n', model.features[1].conv)

    # Calibrate with the training set
    evaluate(data_loader=data_loader_train, model=model, device='cpu', use_amp=False)
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.ao.quantization.convert(model, inplace=True)
    # You may see a user warning about needing to calibrate the model. This warning can be safely ignored.
    # This warning occurs because not all modules are run in each model runs, so some
    # modules may not be calibrated.
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',model.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(model)

    evaluate(data_loader=data_loader_test, model=model, device='cpu', use_amp=False)
    #print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))

def post_training_static_quantization_per_channel(args, model, data_loader_train, data_loader_test):
    per_channel_quantized_model = model
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    per_channel_quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    print(per_channel_quantized_model.qconfig)

    torch.ao.quantization.prepare(per_channel_quantized_model, inplace=True)
    print("Calibration in training set")
    evaluate(data_loader_train, per_channel_quantized_model, 'cpu', False)
    torch.ao.quantization.convert(per_channel_quantized_model, inplace=True)
    print("Converted quantized per channel model")
    evaluate(data_loader_test, per_channel_quantized_model, 'cpu', False)
    torch.jit.save(torch.jit.script(per_channel_quantized_model), "out/"+args.model+args.data_set+"_quantized_per_channel.pth")