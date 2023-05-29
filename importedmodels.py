from fastai.vision.all import *
import streamlit as st
import PIL
import os 

PATH_TO_CWD = os.getcwd()

input_image_file = st.file_uploader("Upload an Input Image",type=['png','jpeg','jpg'])
input_file_path = ''
if input_image_file is not None:
    input_file_path = f'{PATH_TO_CWD}/tmp/input_tmp.{input_image_file.type[-4:]}'
    file_details = {"FileName":input_image_file.name,"FileType":input_image_file.type}
    st.write(file_details)
    img = load_image(input_image_file)
    # st.image(img,height=250,width=250)
    st.image(img)
    # with open(os.path.join("tempDir",image_file.name),"wb") as f: 
    with open(input_file_path,"wb") as f: 
      f.write(input_image_file.getbuffer())         
    st.success("Saved Input Image File")

    style_image_file = st.file_uploader("Upload a Style Image",type=['png','jpeg','jpg'])
    style_file_path = ''
    if style_image_file is not None:
        style_file_path = f'{PATH_TO_CWD}/tmp/style_tmp.{style_image_file.type[-4:]}'
        file_details = {"FileName":style_image_file.name,"FileType":style_image_file.type}
        st.write(file_details)
        img = load_image(style_image_file)
        st.image(img)
        # with open(os.path.join("tempDir",image_file.name),"wb") as f: 
        with open(style_file_path,"wb") as f: 
          f.write(style_image_file.getbuffer())         
        st.success("Saved Style Image File")


        dset = Datasets(style_file_path, tfms=[PILImage.create])
        # dl = dset.dataloaders(after_item=[ToTensor()], after_batch=[IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)], bs=1)
        dl = dset.dataloaders(after_item=[ToTensor()], after_batch=[IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)])
        dl.show_batch(figsize=(7,7))
        style_im = dl.one_batch()[0]

        dset = Datasets(input_file_path, tfms=[PILImage.create])
        # dl = dset.dataloaders(after_item=[ToTensor()], after_batch=[IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)], bs=1)
        dl = dset.dataloaders(after_item=[ToTensor()], after_batch=[IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)])
        dl.show_batch(figsize=(7,7))
        input_im = dl.one_batch()[0]

        learn = load_learner(f'{PATH_TO_CWD}/TrainedModels/uncorrupted.pkl', cpu=False)

        with torch.no_grad():
          res = learn.model(input_im)

        img_np = res.squeeze().permute(1, 2, 0).cpu().numpy()

        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        img_pil = Image.fromarray((img_np * 255).astype('uint8'))
        img_pil.save(f'{PATH_TO_CWD}/generated_images/foo.jpg')
        try:
          st.image(img_pil)
        except:
          image = PIL.Image.open(f'{PATH_TO_CWD}/generated_images/foo.jpg')
          st.image(image, caption='Generated image')
        # img_pil.show()

        # dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
        #   get_items=get_image_files,
        #   splitter=RandomSplitter(0.1, seed=42),
        #   item_tfms=[Resize(224)],
        #   batch_tfms=[Normalize.from_stats(*imagenet_stats)])
        # dls = dblock.dataloaders(Path(input_file_path))

        # feat_net = vgg19(pretrained=True).features.cuda().eval()
        # for p in feat_net.parameters(): p.requries_grad=False

        # layers = [feat_net[i] for i in [1, 6, 11, 20, 29, 22]]; layers

        # _vgg_config = {
        #     'vgg16' : [1, 11, 18, 25, 20],
        #     'vgg19' : [1, 6, 11, 20, 29, 22]
        # }

        # def _get_layers(arch:str, pretrained=True):
        #   "Get the layers and arch for a VGG Model (16 and 19 are supported only)"
        #   feat_net = vgg19(pretrained=pretrained).cuda() if arch.find('9') > 1 else vgg16(pretrained=pretrained).cuda()
        #   config = _vgg_config.get(arch)
        #   features = feat_net.features.cuda().eval()
        #   for p in features.parameters(): p.requires_grad=False
        #   return feat_net, [features[i] for i in config]

        # def get_feats(arch:str, pretrained=True):
        #   "Get the features of an architecture"
        #   feat_net, layers = _get_layers(arch, pretrained)
        #   hooks = hook_outputs(layers, detach=False)
        #   def _inner(x):
        #     feat_net(x)
        #     return hooks.stored
        #   return _inner

        # feats = get_feats('vgg19')

        # class FeatureLoss(Module):
        #   "Combines two losses and features into a useable loss function"
        #   def __init__(self, feats, style_loss, act_loss):
        #     store_attr()
        #     self.reset_metrics()

        #   def forward(self, pred, targ):
        #     # First get the features of our prediction and target
        #     pred_feat, targ_feat = self.feats(pred), self.feats(targ)
        #     # Calculate style and activation loss
        #     style_loss = self.style_loss(pred_feat, targ_feat)
        #     act_loss = self.act_loss(pred_feat, targ_feat)
        #     # Store the loss
        #     self._add_loss(style_loss, act_loss)
        #     # Return the sum
        #     return style_loss + act_loss

        #   def reset_metrics(self):
        #     # Generates a blank metric
        #     self.metrics = dict(style = [], content = [])

        #   def _add_loss(self, style_loss, act_loss):
        #     # Add to our metrics
        #     self.metrics['style'].append(style_loss)
        #     self.metrics['content'].append(act_loss)

        # def act_loss(inp:Tensor, targ:Tensor):
        #   "Calculate the MSE loss of the activation layers"
        #   return F.mse_loss(inp[-1], targ[-1])

        # def get_style_im(img):
        #   dset = Datasets(img, tfms=[PILImage.create])
        #   dl = dset.dataloaders(after_item=[ToTensor()], after_batch=[IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)], bs=1)
        #   return dl.one_batch()[0]

        # def gram(x:Tensor):
        #   "Transpose a tensor based on c,w,h"
        #   n, c, h, w = x.shape
        #   x = x.view(n, c, -1)
        #   return (x @ x.transpose(1, 2))/(c*w*h)

        # im_feats = feats(style_im)

        # im_grams = [gram(f) for f in im_feats]

        # for feat in im_grams:
        #   print(feat.shape)

        # def get_stl_fs(fs): return fs[:-1]

        # def style_loss(inp:Tensor, out_feat:Tensor):
        #   "Calculate style loss, assumes we have `im_grams`"
        #   # Get batch size
        #   bs = inp[0].shape[0]
        #   loss = []
        #   # For every item in our inputs
        #   for y, f in zip(*map(get_stl_fs, [im_grams, inp])):
        #     # Calculate MSE
        #     loss.append(F.mse_loss(y.repeat(bs, 1, 1), gram(f)))
        #   # Multiply their sum by 30000
        #   return 3e5 * sum(loss)

        # loss_func = FeatureLoss(feats, style_loss, act_loss)

        # class ReflectionLayer(Module):
        #     "A series of Reflection Padding followed by a ConvLayer"
        #     def __init__(self, in_channels, out_channels, ks=3, stride=2):
        #         reflection_padding = ks // 2
        #         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        #         self.conv2d = nn.Conv2d(in_channels, out_channels, ks, stride)

        #     def forward(self, x):
        #         out = self.reflection_pad(x)
        #         out = self.conv2d(out)
        #         return out

        # ReflectionLayer(3, 3)

        # class ResidualBlock(Module):
        #     "Two reflection layers and an added activation function with residual"
        #     def __init__(self, channels):
        #           self.conv1 = ReflectionLayer(channels, channels, ks=3, stride=1)
        #           self.in1 = nn.InstanceNorm2d(channels, affine=True)
        #           self.conv2 = ReflectionLayer(channels, channels, ks=3, stride=1)
        #           self.in2 = nn.InstanceNorm2d(channels, affine=True)
        #           self.relu = nn.ReLU()

        #     def forward(self, x):
        #           residual = x
        #           out = self.relu(self.in1(self.conv1(x)))
        #           out = self.in2(self.conv2(out))
        #           out = out + residual
        #           return out

        # class UpsampleConvLayer(Module):
        #     "Upsample with a ReflectionLayer"
        #     def __init__(self, in_channels, out_channels, ks=3, stride=1, upsample=None):
        #         self.upsample = upsample
        #         reflection_padding = ks // 2
        #         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        #         self.conv2d = nn.Conv2d(in_channels, out_channels, ks, stride)

        #     def forward(self, x):
        #         x_in = x
        #         if self.upsample:
        #             x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        #         out = self.reflection_pad(x_in)
        #         out = self.conv2d(out)
        #         return out

        # class TransformerNet(Module):
        #     "A simple network for style transfer"
        #     def __init__(self):
        #         # Initial convolution layers
        #         self.conv1 = ReflectionLayer(3, 32, ks=9, stride=1)
        #         self.in1 = nn.InstanceNorm2d(32, affine=True)
        #         self.conv2 = ReflectionLayer(32, 64, ks=3, stride=2)
        #         self.in2 = nn.InstanceNorm2d(64, affine=True)
        #         self.conv3 = ReflectionLayer(64, 128, ks=3, stride=2)
        #         self.in3 = nn.InstanceNorm2d(128, affine=True)
        #         # Residual layers
        #         self.res1 = ResidualBlock(128)
        #         self.res2 = ResidualBlock(128)
        #         self.res3 = ResidualBlock(128)
        #         self.res4 = ResidualBlock(128)
        #         self.res5 = ResidualBlock(128)
        #         # Upsampling Layers
        #         self.deconv1 = UpsampleConvLayer(128, 64, ks=3, stride=1, upsample=2)
        #         self.in4 = nn.InstanceNorm2d(64, affine=True)
        #         self.deconv2 = UpsampleConvLayer(64, 32, ks=3, stride=1, upsample=2)
        #         self.in5 = nn.InstanceNorm2d(32, affine=True)
        #         self.deconv3 = ReflectionLayer(32, 3, ks=9, stride=1)
        #         # Non-linearities
        #         self.relu = nn.ReLU()

        #     def forward(self, X):
        #         y = self.relu(self.in1(self.conv1(X)))
        #         y = self.relu(self.in2(self.conv2(y)))
        #         y = self.relu(self.in3(self.conv3(y)))
        #         y = self.res1(y)
        #         y = self.res2(y)
        #         y = self.res3(y)
        #         y = self.res4(y)
        #         y = self.res5(y)
        #         y = self.relu(self.in4(self.deconv1(y)))
        #         y = self.relu(self.in5(self.deconv2(y)))
        #         y = self.deconv3(y)
        #         return y

        # net = TransformerNet()

        # learn = Learner(dls, TransformerNet(), loss_func=loss_func)

        # learn.load_model('/Users/reidwilson/workspace/Capstone/TrainedModels/Alfred_Sisley.pth', cpu=False)
        # learn = load_learner('./TrainedModels/Alfred_Sisley.pth', cpu=False)


        # print(learn)