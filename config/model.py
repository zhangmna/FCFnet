import torch
from torch import nn
from torch import functional as F
from torch.autograd import Variable
from torchvision.models import vgg19
import settings
class SEBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()    
        mid = int(input_dim /4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, input_dim),
            nn.Sigmoid()
        ) 

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class NoSEBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

    def forward(self, x):
        return x

SE = SEBlock if settings.use_se else NoSEBlock
class Res(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(Res, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel, out_channel,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channel, out_channel,3,1,1),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        out=x+self.conv(x)
        return out
class Tradition(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(Tradition, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel, out_channel,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channel, out_channel,3,1,1),
            nn.LeakyReLU(0.2))
    def forward(self, x):
        out=self.conv(x)
        return out  
class Dilation(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(Dilation, self).__init__()
        self.path1=nn.Sequential(nn.Conv2d(in_channel, out_channel,3,1,1,1),nn.LeakyReLU(0.2))
        self.path3 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 3, 3), nn.LeakyReLU(0.2))
        self.path5 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 5, 5), nn.LeakyReLU(0.2))
    def forward(self, x):
        path1 = self.path1(x)
        path3 = self.path3(x)
        path5 = self.path5(x)
        out=path1+path3+path5
        return out
class ConvGRU(nn.Module):
    def __init__(self, inp_dim, oup_dim):
        super().__init__()
        self.conv_xz = nn.Conv2d(inp_dim, oup_dim, 3,1,1)
        self.conv_xr = nn.Conv2d(inp_dim, oup_dim, 3,1,1)
        self.conv_xn = nn.Conv2d(inp_dim, oup_dim, 3,1,1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.se = SE(oup_dim)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        z = self.sigmoid(self.conv_xz(x))
        f = self.tanh(self.conv_xn(x))
        h = z * f
        h = self.relu(self.se(h))
        return h
class Compact(nn.Module):  #紧凑的，紧密的
    def __init__(self,in_channel, out_channel):
        super(Compact, self).__init__()
        self.df1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,3,1,1,1),nn.BatchNorm2d(out_channel),nn.LeakyReLU(0.2))
        self.conv1_1 = nn.Conv2d(2*out_channel, out_channel, 1, 1)
        self.conv1_2 = nn.Conv2d(2*out_channel, out_channel, 3, 1, 1)
        self.df3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 3, 3),nn.BatchNorm2d(out_channel), nn.LeakyReLU(0.2))
        self.df5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 5, 5), nn.BatchNorm2d(out_channel),nn.LeakyReLU(0.2))
        self.conv1_3 = nn.Sequential(nn.Conv2d(3*out_channel,out_channel,1,1),SEBlock(out_channel))
    def forward(self, x):
        out_df1=self.df1(x)
        conv1_1=self.conv1_1(torch.cat([x,out_df1],dim=1))
        conv1_2=self.conv1_2(torch.cat([x,out_df1],dim=1))
        df3 = self.df3(conv1_1)
        df5 = self.df5(conv1_2)
        compact=self.conv1_3(torch.cat([out_df1,df3,df5],dim=1))
        return compact
class MSDC(nn.Module):
    def __init__(self, mid_channel,out_channel):
        super(MSDC, self).__init__()

        if settings.dilation is True:
            self.path1 = nn.Sequential(
                nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, 1),
                nn.LeakyReLU(0.2),
                # nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, 1),
                # nn.LeakyReLU(0.2)
            )
            self.path3 = nn.Sequential(
                nn.Conv2d(mid_channel, mid_channel, 3, 1, 3, 3),
                nn.LeakyReLU(0.2),
                # nn.Conv2d(mid_channel, mid_channel, 3, 1, 3, 3),
                # nn.LeakyReLU(0.2)
            )
            self.path5 = nn.Sequential(
                nn.Conv2d(mid_channel, mid_channel, 3, 1, 5, 5),
                nn.LeakyReLU(0.2),
                # nn.Conv2d(mid_channel, mid_channel, 3, 1, 5, 5),
                # nn.LeakyReLU(0.2)
            )

            self.cat1_3 = nn.Sequential(
                nn.Conv2d(2 * mid_channel, mid_channel, 1, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, 1),
                nn.LeakyReLU(0.2)
            )
            self.cat3_5 = nn.Sequential(
                nn.Conv2d(2 * mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, 1), nn.LeakyReLU(0.2)
            )
            self.cat_final = nn.Sequential(
                nn.Conv2d(2 * mid_channel, out_channel, 1, 1),
                nn.LeakyReLU(0.2),
                SE(out_channel)
            )
        else:
            self.path1 = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, 1), nn.LeakyReLU(0.2),
                # nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, 1),
                # nn.LeakyReLU(0.2)
            )
            self.path3 = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, 1), nn.LeakyReLU(0.2),
                # nn.Conv2d(mid_channel, mid_channel, 3, 1, 3, 3),
                # nn.LeakyReLU(0.2)
            )
            self.path5 = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, 1), nn.LeakyReLU(0.2),
                # nn.Conv2d(mid_channel, mid_channel, 3, 1, 5, 5),
                # nn.LeakyReLU(0.2)
            )

            self.cat1_3 = nn.Sequential(nn.Conv2d(2 * mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, 1), nn.LeakyReLU(0.2))
            self.cat3_5 = nn.Sequential(nn.Conv2d(2 * mid_channel, mid_channel, 1, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, 1), nn.LeakyReLU(0.2))
            self.cat_final = nn.Sequential(nn.Conv2d(2 * mid_channel, out_channel, 1, 1), nn.LeakyReLU(0.2),
                SE(out_channel))

    def forward(self, x):
        path1 = self.path1(x)
        path3 = self.path3(x)
        path5 = self.path5(x)
        cat1_3 = self.cat1_3(torch.cat([path1, path3], dim=1))
        cat3_5 = self.cat3_5(torch.cat([path3, path5], dim=1))
        final = self.cat_final(torch.cat([cat1_3, cat3_5], dim=1))
        return final
class Dense_dilation(nn.Module):  # DCDCB   densely connected dilation convolution block
    def __init__(self,mid_channel,out_channel):
        super(Dense_dilation, self).__init__()
        self.num=settings.num_dense_dilation
        self.mid_channel = settings.mid_channel
        self.cat_dense=nn.ModuleList()
        self.dense_conv=nn.ModuleList()
        for i in range(self.num):    
            self.cat_dense.append(nn.Conv2d((i+2)*self.mid_channel,self.mid_channel,1,1)) # 密集连接
        for i  in range(self.num):
            self.dense_conv.append(nn.Sequential(nn.Conv2d(self.mid_channel,self.mid_channel,3,1,2**i,2**i),nn.LeakyReLU(0.2)))  # 扩张卷积，保证输入和输出相等，padding和dilation是相等的。
        self.out=nn.Conv2d(self.mid_channel,3,1,1)
    def forward(self, x):
        img = []
        img.append(x)
        for i in range(self.num):
            x=self.dense_conv[i](x)
            img.append(x)
            x=self.cat_dense[i](torch.cat(img, dim=1))
        x=self.out(x)
        return x
class My_unit(nn.Module):  # 小的网络block
    def __init__(self, mid_channel,out_channel):
        super(My_unit, self).__init__()
        self.convert = nn.Sequential(
            nn.Conv2d(3, mid_channel, 3, 1, 1, 1), nn.LeakyReLU(0.2),
        )
        self.Dense_dilation=Dense_dilation(mid_channel,out_channel)
    def forward(self, x):
        convert=self.convert(x)
        Dense_dilation=self.Dense_dilation(convert)
        return Dense_dilation
Unit = {
    'traditional': Tradition,
    'res': Res,
    'dilation': Dilation,
    'GRU': ConvGRU,
    'my_unit':My_unit
}[settings.unit]

class My_blocks(nn.Module):     # 大的网络框架
    def __init__(self,in_channel):
        super(My_blocks, self).__init__()
        self.in_channel = 3
        self.mid_channel = settings.mid_channel   # mid_channel 16
        self.num = settings.block_num    # num 12
        self.res = nn.ModuleList()  
        self.cat_1 = nn.ModuleList()
        self.cat_2 = nn.ModuleList()
        self.cat_dense=nn.ModuleList()
        if settings.connection_style == 'dense_connection':  
            for i in range(self.num+1):              
                self.cat_dense.append(nn.Conv2d((i+2)*self.in_channel,3,1,1))  # 密集连接的实现，输出通道数是3
        self.msdc=My_unit(settings.mid_channel, 3)
    def forward(self, next_img,rainyimg):
        if settings.connection_style == 'dense_connection':
            img = []
            rain = []
            img.append(next_img)
            for i in range(self.num):
                rain_streak=self.msdc(next_img)
                rain.append(rain_streak)
                next_img=rainyimg-rain_streak   # 网络学出来的无雨图
                img.append(next_img)
                next_img=self.cat_dense[i](torch.cat(img, dim=1))
            return rain
        # if settings.connection_style == 'multi_short_skip_connection':
        #     out=[]
        #     out.append(x)
        #     for i in range(self.num):
        #         x=self.res[i](x)
        #         out.append(x)
        #         if i%2==0 & i>=2:
        #             odd=[] #odd：奇数
        #             for j in range(i):
        #                 odd.append(out[2*j+1])
        #             x=self.cat_1[int((i-2)/2)](torch.cat(odd,dim=1))
        #         if i%2==1:
        #             even=[]
        #             even.append(out[0])
        #             even.append(out[2])
        #             if i>=3:
        #                 for s in range(int((i-1)/2)):
        #                     even.append(out[2*(s+2)])
        #             x=self.cat_2[int((i-1)/2)](torch.cat(even,dim=1))
        #     return x
        # elif settings.connection_style == 'symmetric_connection':
        #     out=[]
        #     out.append(x)
        #     for i in range(self.num):
        #         x=self.res[i](x)
        #         out.append(x)
        #         if i >= (int(self.num/2)):
        #             x=self.cat_2[int(i-int(self.num/2))](torch.cat([out[-1],out[(-2)*(i-int(self.num/2)+1)-1]],dim=1))
        #     return x
        # elif settings.connection_style == 'no_connection':
        #     for i in range(self.num):
        #         x=self.res[i](x)
        #     return x
class Refinement(nn.Module):
    def __init__(self):
        super(Refinement, self).__init__()
        self.channel = settings.mid_channel
        self.refinement = nn.Sequential(
            nn.Conv2d(3, self.channel, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channel, self.channel, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channel, 3, 1, 1)
        )
    def forward(self, x):
        out = self.refinement(x)
        return out
class RESCAN(nn.Module):
    def __init__(self):
        super(RESCAN, self).__init__()
        channel_num = settings.mid_channel
        self.extract = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.LeakyReLU(0.2),
            SE(channel_num)
        )
        self.dense = My_blocks(channel_num)
        self.refinement = Refinement()
    def forward(self, x):
        extract=self.extract(x)
        rain = self.dense(extract,x)   #rain是一个列表
        refinement = self.refinement(x-rain[-1])
        return rain,refinement  # rain表示的是雨条信息，refinement表示的是去雨后的结果。
class VGG(nn.Module):
    'Pretrained VGG-19 model features.'
    def __init__(self, layers=(3, 6, 8, 11), replace_pooling = False):
        super(VGG, self).__init__()
        self.layers = layers
        self.instance_normalization = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU()
        self.model = vgg19(pretrained=True).features
        # Changing Max Pooling to Average Pooling
        if replace_pooling:
            self.model._modules['4'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['9'] = nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['18'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['27'] =nn.AvgPool2d((2,2), (2,2), (1,1))
            self.model._modules['36'] = nn.AvgPool2d((2,2), (2,2), (1,1))
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in enumerate(self.model):
            x = layer(x)
            if name in self.layers:
                features.append(x)
                if len(features) == len(self.layers):
                    break
        return features

#ts = torch.Tensor(16, 3, 64, 64).cuda()
#vr = Variable(ts)
#net = RESCAN().cuda()
#print(net)
#oups = net(vr)
#for oup in oups:
#    print(oup.size())

if __name__ == '__main__':
    x = My_blocks(3)
    print(x)