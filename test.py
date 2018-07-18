import tensorflow as tf

tf.reset_default_graph()

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./20180204-160909/model-20180204-160909.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./20180204-160909'))
    graph = tf.get_default_graph()
    print(graph)
    collect_keys = graph.get_all_collection_keys()
    # for k in collect_keys:
    #     print(graph.get_collection(k))
    # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    # print(embeddings)
    # print(sess.graph)
    # file_writer = tf.summary.FileWriter('./logs/', sess.graph)

    print('------------------------------------------------------')
    for var in tf.global_variables():
        print('all variables: ' + var.op.name)
    
