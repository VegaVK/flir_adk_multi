import rosbag


# from std_msgs.msg import Int32, String





thermal_bag = rosbag.Bag('bag2_2020-02-12-11-33-31.bag')
zed_bag = rosbag.Bag('bag2_2020-02-12-11-33-35.bag')

flir1 = []
flir2 = []
flir3 = []
flir4 = []
flir5 = []
zed = []
objects = []
i=0


for flir, ther_image, t1 in thermal_bag.read_messages(topics=['/flir_boson1/image_rect']):
	flir1.append([t1,ther_image])
# thermal_bag.close()

for flir, ther_image, t1 in thermal_bag.read_messages(topics=['/flir_boson2/image_rect']):
	flir2.append([t1,ther_image])
#thermal_bag.close()

for flir, ther_image, t1 in thermal_bag.read_messages(topics=['/flir_boson3/image_rect']):
	flir3.append([t1,ther_image])
#thermal_bag.close()

for flir, ther_image, t1 in thermal_bag.read_messages(topics=['/flir_boson4/image_rect']):
	flir4.append([t1,ther_image])
#thermal_bag.close()

for flir, ther_image, t1 in thermal_bag.read_messages(topics=['/flir_boson5/image_rect']):
	flir5.append([t1,ther_image])
thermal_bag.close()

for zed_node, zed_image, t2 in zed_bag.read_messages(topics=['/zed/zed_node/left/image_rect_color']):
	zed.append([t2,zed_image])  

for radar_node, radar_data, t2 in zed_bag.read_messages(topics=['/as_tx/objects']):
	objects.append([t2,radar_data])  
zed_bag.close()	


bag = rosbag.Bag('bag2_2020-02-12-11-33-Combo.bag', 'w')
for i in range(max(len(flir1),len(flir2),len(flir3),len(flir4),len(flir5))):
	if i<len(flir1):
		bag.write('flir_boson1/image_rect', flir1[i][1], flir1[i][0])
	if i<len(flir2):
		bag.write('flir_boson2/image_rect', flir2[i][1], flir2[i][0])
	if i<len(flir3):
		bag.write('flir_boson3/image_rect', flir3[i][1], flir3[i][0])
	if i<len(flir4):
		bag.write('flir_boson4/image_rect', flir4[i][1], flir4[i][0])
	if i<len(flir5):
		bag.write('flir_boson5/image_rect', flir5[i][1], flir5[i][0])
	if i<len(zed):
		bag.write('/zed/zed_node/left/image_rect_color', zed[i][1], zed[i][0])
	if i<len(objects):
		bag.write('/as_tx/objects', objects[i][1], objects[i][0])
		
bag.close()
