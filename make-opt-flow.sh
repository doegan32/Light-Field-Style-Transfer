# Specify the path to the optical flow utility here.
# Also check line 44 and 47 whether the arguments are in the correct order.

# deepflow and deepmatching optical flow binaries
flowCommandLine="bash run-deepflow.sh"

if [ -z "$flowCommandLine" ]; then
  echo "Please open make-opt-flow.sh and specify the command line for computing the optical flow."
  exit 1
fi

if [ ! -f ./consistencyChecker/consistencyChecker ]; then
  if [ ! -f ./consistencyChecker/Makefile ]; then
    echo "Consistency checker makefile not found."
    exit 1
  fi
  cd consistencyChecker/
  make
  cd ..
fi

filePattern=$1
folderName=$2
s1=$3
t1=$4
s2=$5
t2=$6


if [ "$#" -le 1 ]; then
   echo "Usage: ./make-opt-flow <filePattern> <outputFolder> <viewpoint1> <viewpoint2>"
   echo -e "\tfilePattern:\tFilename pattern of the frames of the videos."
   echo -e "\toutputFolder:\tOutput folder."
   echo -e "\tviewpoint1:\tindex of first viewpoint"
   echo -e "\tviewpoint2:\tindex of second viewpoint"
   exit 1
fi

mkdir -p "${folderName}"


file1=$(printf "$filePattern" "$s2" "$t2")
file2=$(printf "$filePattern" "$s1" "$t1")


eval $flowCommandLine "$file1" "$file2" "${folderName}/forward_${s2}_${t2}_${s1}_${t1}.flo"
eval $flowCommandLine "$file2" "$file1" "${folderName}/backward_${s1}_${t1}_${s2}_${t2}.flo"
./consistencyChecker/consistencyChecker "${folderName}/backward_${s1}_${t1}_${s2}_${t2}.flo" "${folderName}/forward_${s2}_${t2}_${s1}_${t1}.flo" "${folderName}/reliable_${s1}_${t1}_${s2}_${t2}.txt"
./consistencyChecker/consistencyChecker "${folderName}/forward_${s2}_${t2}_${s1}_${t1}.flo" "${folderName}/backward_${s1}_${t1}_${s2}_${t2}.flo" "${folderName}/reliable_${s2}_${t2}_${s1}_${t1}.txt"

