import tensorflow as tf
import os
model_dir = './models/saved_model.pb'
def create_graph():
    with tf.io.gfile.GFile(model_dir, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

create_graph()
tensor_name_list = [tensor.name for tensor in tf.compat.v1.get_default_graph().as_graph_def().node]
print("input_name:"+tensor_name_list[0])
print("output_name:"+str(tensor_name_list))
