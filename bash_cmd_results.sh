# Incremental Update
printf "#######################################################################\nIU results:\n#"
cat MovieLens/IU/ckpts/IU_train11-23_test24-30_1epoch_0.001/period30/test_metrics.txt
printf "\n"

# Batch Update 3
printf "#######################################################################\nBU-3 results:\n#"
cat MovieLens/BU/ckpts/BU_train11-23_test24-30_3_1epoch_0.001/period30/test_metrics.txt
printf "\n"

# Batch Update 5
printf "#######################################################################\nBU-5 results:\n#"
cat MovieLens/BU/ckpts/BU_train11-23_test24-30_5_1epoch_0.001/period30/test_metrics.txt
printf "\n"

# Batch Update 7
printf "#######################################################################\nBU-7 results:\n#"
cat MovieLens/BU/ckpts/BU_train11-23_test24-30_7_1epoch_0.001/period30/test_metrics.txt
printf "\n"

# SMLmf
printf "#######################################################################\nSMLmf results:\n#"
cat MovieLens/SMLmf/SML/ckpts/SMLmf_train11-23_test24-30_1epoch_1epoch_0.01_0.01/period30/test_metrics.txt
printf "\n"


