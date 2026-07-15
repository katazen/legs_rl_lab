echo "start copy tty_imu.rules to /etc/udev/rules.d/"
sudo cp tty_imu.rules /etc/udev/rules.d

service udev reload
sleep 2
service udev restart
echo "Finish!!!"
