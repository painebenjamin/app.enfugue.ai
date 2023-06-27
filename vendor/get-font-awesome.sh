#!/usr/bin/env bash
VERSION="6.2.0"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [ $# -eq 1 ]; then
    ROOT=$1
else 
    ROOT=$DIR/../src
fi
FA_DIRNAME=fontawesome-free-$VERSION-web

FONTS=$ROOT/fonts
FONTS_TARGET=$FONTS/vendor/fa
FONTS_FILES=( fa-brands-400 fa-regular-400 fa-solid-900 )
FONTS_FORMATS=( ttf woff2 )

CSS=$ROOT/css
CSS_TARGET=$CSS/vendor/fa
CSS_FILES=( brands.min.css fontawesome.min.css regular.min.css solid.min.css )

cd $DIR
wget https://use.fontawesome.com/releases/v$VERSION/$FA_DIRNAME.zip
unzip $FA_DIRNAME.zip

rm -rf $CSS_TARGET
mkdir -p $CSS_TARGET
for CSS_FILE in "${CSS_FILES[@]}"; do
    cp $FA_DIRNAME/css/$CSS_FILE $CSS_TARGET/
    sed -i -e 's/..\/webfonts\//..\/..\/..\/fonts\/vendor\/fa\//g' $CSS_TARGET/$CSS_FILE
done

rm -rf $FONTS_TARGET
mkdir -p $FONTS_TARGET
for FONTS_FILE in "${FONTS_FILES[@]}"; do
    for FONTS_FORMAT in "${FONTS_FORMATS[@]}"; do
        cp $FA_DIRNAME/webfonts/$FONTS_FILE.$FONTS_FORMAT $FONTS_TARGET/
    done
done

rm -rf $DIR/$FA_DIRNAME.zip $DIR/$FA_DIRNAME

echo "Successfully retrieved FontAwesome Free $VERSION"
