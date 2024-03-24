mkdir coco
cd coco

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm -rf val2017.zip
python ../3channels.py val2017
zip -r val2017.zip val2017

mkdir train2017
zip -r train2017.zip train2017
rmdir train2017

mkdir test2017
zip -r test2017.zip test2017
rmdir test2017

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip