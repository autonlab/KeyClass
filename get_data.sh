mkdir data/
cd data/

green=`tput setaf 2`
reset=`tput sgr0`

mkdir agnews/
cd agnews/
echo ${green}===Downloading AG News Data...===${reset}
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/a/illinois.edu/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/a/illinois.edu/uc?export=download&id=1zszTJudS8RMgTQxURkt1w2MhswNGA6Oa' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zszTJudS8RMgTQxURkt1w2MhswNGA6Oa" -O agnews.zip && rm -rf /tmp/cookies.txt
echo ${green}===Unzipping AG News Data...===${reset}
unzip agnews.zip && rm agnews.zip
cd ../

mkdir amazon/
cd amazon/
echo ${green}===Downloading Amazon Data...===${reset}
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/a/illinois.edu/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/a/illinois.edu/uc?export=download&id=1pRt5mPuuVbi-ZXD8QZzw_7DlAnEg3X15' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pRt5mPuuVbi-ZXD8QZzw_7DlAnEg3X15" -O amazon.zip && rm -rf /tmp/cookies.txt
echo ${green}===Unzipping Amazon Data...===${reset}
unzip amazon.zip && rm amazon.zip
cd ../

mkdir dbpedia/
cd dbpedia/
echo ${green}===Downloading DBPedia Data...===${reset}
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/a/illinois.edu/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/a/illinois.edu/uc?export=download&id=1nCQQAC6XwfnyKtzWlNElMtz4s12kxfe7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nCQQAC6XwfnyKtzWlNElMtz4s12kxfe7" -O dbpedia.zip && rm -rf /tmp/cookies.txt
echo ${green}===Unzipping DBPedia Data...===${reset}
unzip dbpedia.zip && rm dbpedia.zip
cd ../

mkdir imdb/
cd imdb/
echo ${green}===Downloading IMDB Data...===${reset}
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/a/illinois.edu/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/a/illinois.edu/uc?export=download&id=1c8X_Ooth2fQleCVz2gCXlOd3-zzE9Mws' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c8X_Ooth2fQleCVz2gCXlOd3-zzE9Mws" -O imdb.zip && rm -rf /tmp/cookies.txt
echo ${green}===Unzipping IMDB Data...===${reset}
unzip imdb.zip && rm imdb.zip
cd ../
