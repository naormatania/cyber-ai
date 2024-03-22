mkdir coco
cd coco

wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip

mkdir train2017
zip -r train2017.zip train2017
rmdir train2017

mkdir val2017
zip -r val2017.zip val2017
rmdir val2017

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip