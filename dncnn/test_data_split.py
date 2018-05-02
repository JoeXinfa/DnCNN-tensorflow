
"""
Test split data batches to each worker/task.
"""

# Simplest code, but issue is that the last three batches in each epoch
# always on workers 1-3, deterministically, which accumulates and slows.
# task_count, batch_count =  9, 21
# for batch_id in range(0, batch_count, task_count):
#     for task_index in range(task_count):
#         current_batch = batch_id + task_index + 1
#         if current_batch <= batch_count:
#             print(current_batch, task_index)

task_count, batch_count = 20, 11
# task_count, batch_count =  9, 21
# task_count, batch_count =  2, 21
if batch_count > task_count:
    batch_stop = batch_count - task_count
    batch_drop = batch_count % task_count
    print("WARNING: the last %d batches are unused." % batch_drop)
else:
    batch_stop = 1
for batch_id in range(0, batch_stop, task_count):
    for task_index in range(task_count):
        current_batch = batch_id + task_index + 1
        if current_batch <= batch_count:
            print(current_batch, task_index)
