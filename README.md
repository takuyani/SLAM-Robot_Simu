# SLAM-Robot_Simu

SLAM-Robot開発用のシミュレーション

## 説明
__SLAM-Robot_Simu__ はSLAM-Robot開発のためのシミュレーション・リポジトリです。シミュレーション用のスクリプトは基本的にPythonで書いてます。

***DEMO:***  
[![DEMO](http://img.youtube.com/vi/7Lxlnb39wtU/0.jpg)](http://www.youtube.com/watch?v=7Lxlnb39wtU)

## リポジトリ構成
以下、リポジトリ構成です。
* __extended_kalman_filter.py__　⇒拡張カルマンフィルタ
* __mylib/__  ⇒共通機能
    * __\_\_init\_\_.py__　⇒パッケージ初期化
    * __error_ellipse.py__　⇒誤差楕円生成
    * __transform.py__　⇒座標変換

## 環境
私が本シミュレーションを実施している環境になります。
- OS: windows 10 64bit
- 統合開発環境：　Eclipse 4.6 Neon 64bit (Pleiades)
- Python環境：　Anaconda 4.2.0 64bit (Python 3.5 version)

---
## 環境構築
### Anacondaのインストール
1. 以下のサイトからAnacondaモジュールをダウンロードする。現在ではPyhon2系とPyhon3系が選べるようだが、3系にすることを推奨する。  
https://store.continuum.io/cshop/anaconda/

1. ダウンロードされたインストーラーを実行し、インストーラー画面に従い、インストールを完了させる。

### Eclipse(Pleiades)のインストール
1. 以下のサイトからEclipse(Pleiades)モジュールをダウンロードする。  
http://mergedoc.osdn.jp/

2. ダウンロードされたZipファイルを適当な場所に解凍する。

### AnacondaとEclipseの連携設定
1. Eclipseを起動し、「メニュー」⇒「ウィンドウ」⇒「設定」を選択する。
![fig.conf1](https://c1.staticflickr.com/1/270/32072426735_259b16ff22_b.jpg)

2. 左のペインから「PyDev」⇒「インタープリター」⇒「Python Interpreter」を選択し、右のペインの「Python インタープリター」で、右上のボタン「新規」をクリックする。
![fig.conf2](https://c1.staticflickr.com/1/309/31261885243_8995e30141_b.jpg)

3. 「インタープリター選択のダイアログ」が立ち上がるので以下ように入力する。  終わったら「OK」をクリックする。
 - __インタープリター名：__  
 ⇒Anaconda Python（任意で構わない）
 - __インタープリター実行可能ファイル：__  
 ⇒[Anacondaのインストール](#Anacondaのインストール)でインストールしたディレクトリに「python.exe」があるのでそれを選択する。デフォルトでは「C:\Program Files\Anaconda3\python.exe」。

 ![fig.conf3](https://c1.staticflickr.com/1/591/31954397401_aa4207bc03_z.jpg)

4. そのまま「OK」をクリックする。  
![fig.conf4](https://c1.staticflickr.com/1/512/31924021842_ce709db53f_z.jpg)

5. 右のペイン(Python インタープリター)で、上部のリストに「Anaconda Python」という項目が追加されているのを確認したら、右側のボタン「上へ」をクリックして、リスト項目「Anaconda Python」をリストの先頭に移動させる。移動が完了したら、画面右下のボタン「適用」をクリックした後に「OK」をクリックする。
![fig.conf5](https://c1.staticflickr.com/1/608/32072426855_9471ddcc91_b.jpg)

---
## アニメーションの保存設定
スクリプトによっては以下のようにアニメーションを保存できるものが存在する。  
![fig.anim1](https://c2.staticflickr.com/6/5611/32073283135_8988f828a6_z.jpg)  
アニメーションを保存するには以下の設定が必要です。

### ffmpegのインストール
1. 以下のサイトからffmpegモジュールをダウンロードする。  
https://ffmpeg.zeranoe.com/builds/

2. ダウンロードされたZipファイルを適当な場所に解凍する。

### 環境変数の設定
1. 環境変数に「ffmpeg.exe」が存在するディレクトリを指定します。  
（例）
C:\Apl\ffmpeg-20161230-6993bb4-win64-static\ffmpeg-20161230-6993bb4-win64-static\bin
