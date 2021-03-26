from torch.utils import data
class Load_Mnist(data.Dataset):
    def __init__(self, data, target, transform):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

def Mnist_processor(args, training_data, training_label, testing_data, testing_label):
    min_max = [(-0.8826567065619495, 9.001545489292527),
                (-0.6661464580883915, 20.108062262467364),
                (-0.7820454743183202, 11.665100841080346),
                (-0.7645772083211267, 12.895051191467457),
                (-0.7253923114302238, 12.683235701611533),
                (-0.7698501867861425, 13.103278415430502),
                (-0.778418217980696, 10.457837397569108),
                (-0.7129780970522351, 12.057777597673047),
                (-0.8280402650205075, 10.581538445782988),
                (-0.7369959242164307, 10.697039838804978)]

    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize([min_max[args.normal_class][0]],
                                                        [min_max[args.normal_class][1] \
                                                        -min_max[args.normal_class][0]])])
    data_train = Load_Mnist(training_data, training_label, transform)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, 
                                shuffle=True, num_workers=0)
    data_test = Load_Mnist(testing_data, testing_label, transform)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size, 
                                shuffle=True, num_workers=0)
    return dataloader_train, dataloader_test