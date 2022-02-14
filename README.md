# 🕵️‍♂️ People Segmentation

A project to segment people on the image using the U-Net Deep Learning algorithm.

![segmentation](https://user-images.githubusercontent.com/70326958/153869558-ecb34f05-ebcf-41b3-9783-d7d7b6dc3a91.jpg)

## 🛠️ Installation
Install project with the following commands:

```sh
git pull https://github.com/xcapt0/people_segmentation.git
cd people_segmentation
mkdir tmp
docker build -t segmentation .
docker run --rm -it -v $(pwd)/tmp:/app/tmp segmentation
```

## 🔍 Usage

`--segment` – path where images are stored. Pass the folder `path/to/images` or the file `path/to/image.jpg`

`--checkpoint` – path to the model checkpoint `path/to/model.pth`

`--save_dir` – directory where segmented images will be stored. Pass new folder `example`. If there is no folder or if passed `example/people` 
then folders will be created recursively

Run the script with the following command:

```sh
python segmentation.py --segment images \ 
                       --checkpoint weights/model.pth \
                       --save_dir segmented/people
```

## 📝 License

Copyright © 2022 [xcapt0](https://github.com/xcapt0).<br />
This project is [MIT](https://github.com/xcapt0/people_segmentation/blob/main/LICENSE) licensed.
