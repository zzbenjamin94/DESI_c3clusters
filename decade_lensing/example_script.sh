pythonfile=/home/zwshao/Codes/arsenal/shaopy/shaopy/lensing/measurement/precompute.py

catdir="/home/zwshao/Data/DESI/Y1/BGS/SMHM_cat/Y1LSScat_zoumstar/DECADEnoHSC"
resdir="/home/zwshao/Results/BGS_SMHM/Zu2015_mcuts/Y1LSScat_zoumstar_lensing/DECADEnoHSC"

for file in `find ${catdir} -maxdepth 1 -name "DECADEnoHSC_BGS_BRIGHT*_lssweight_finemsbin.hdf5" | sort -n`
do
python $pythonfile -n 80 --fname $file --dz 0.1 --bintype 'log' --rmin 0.01 --rmax 30 --nbins 10 --withh --weight 'wtot' --source decade --jkname 'jkid'
done

for file in `find ${catdir} -maxdepth 1 -name 'DECADEnoHSC_rand_BGS_BRIGHT*_lssweight_finemsbin.hdf5' | sort -n`
do
python $pythonfile -n 80 --fname $file --dz 0.1 --bintype 'log' --rmin 0.01 --rmax 30 --nbins 10 --withh --weight 'wtot' --source decade --jkname 'jkid'
done

mv $catdir/*bin=log* $resdir

pythonfile=/home/zwshao/Codes/arsenal/shaopy/shaopy/lensing/measurement/post_process.py
filepath=$resdir
for file in `find $filepath -maxdepth 1 -name '*lssweight*decade*dz=*_bin=log.hdf5' ! -name "DECADEnoHSC_rand*"`
do
dfile=${file##*\/}
random=${dfile/DECADEnoHSC/DECADEnoHSC_rand}
outname=${dfile/\.hdf5_preres/}
outname=${outname/hdf5/ecsv}
python $pythonfile -l ${filepath}/${dfile} -r ${filepath}/${random} -o ${filepath}/${outname} --jkname 'jkid' --source decade --ncpu 80
done
