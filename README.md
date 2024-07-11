sudo docker build -t foto_script .

sudo docker run -d \
    -v /mnt/nas/foto/sken-in:/mnt/nas/foto/sken-in \
    -v /mnt/nas/foto/sken-out:/mnt/nas/foto/sken-out \
    -v /home/root-fame/foto_script/:/app \
    --name foto_script \
    foto_script

sudo crontab -e #docker start foto_script