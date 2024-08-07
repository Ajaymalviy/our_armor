import vlc
import time
player=vlc.MediaPlayer('rtsp://KD124209226093.ppp-bb.dion.ne.jp:554/live.sdp')

player.play()
while 1:
    time.sleep(1)
    player.video_take_snapshot(0, '.snapshot.tmp.png', 0, 0)
