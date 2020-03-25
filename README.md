**Introduction**

The Project detects objects such as *Safety Helmets*, *Safety Vests* along with a *Person* using Mask-RCNN. Pre-trained weights from the COCO Dataset are used and the images are annotated in COCO format.
The project works by adding 2 new classes i.e Safety Helmets and Safety Vests onto the coco dataset which is already trained for the Person class.
The training set was evenly distributed among all the 3 classes, with there being around 730 annotations for each class.

**How Do I use a pre-trained class and further train new classes?**

The images in your custom dataset should be annotated for all the classes to be detected. Using ```self.add_class(source_name, class_id, class_name)``` the new classes are added. Make sure ```NUM_CLASSES = 1 + x``` is set based upon the total classes to be detected (Background+Total Classes);here ```NUM_CLASSES = 1 + 3```.

**Can this model further be improved?**

I haven't tuned the model much, so there is a whole lot that can be improved. You can further augment your images, set the loss weights and tune other several hyperparameters.

**Annotating Images**

My requirement was just detection and not segmentation, however I did want to accomplish this using Mask-RCNN and not YOLO, Faster-RCNN, etc. I followed [this](https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/) great tutorial for setting up my dataset in coco format.

**System Configuration**

I used *Google Colab* with 25GB RAM and 68GB GB Disk.

**Test Results**

![Test_image1](/Test_Set/t0.png)

![Test_image2](/Test_Set/t1.png)

![Test_image3](/Test_Set/t2.png)

![Test_image4](/Test_Set/t3.png)
