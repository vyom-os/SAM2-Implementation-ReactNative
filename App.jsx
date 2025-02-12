import React, { useState, useEffect } from 'react';
import { Image, View, StyleSheet, Text, TouchableOpacity, ActivityIndicator } from 'react-native';
import { NativeModules } from 'react-native';
import * as ImagePicker from 'react-native-image-picker';
import * as RNFS from 'react-native-fs';

const { SAMModule } = NativeModules;
const DISPLAY_WIDTH = 400;
const DISPLAY_HEIGHT = 400

const App = () => {
    const [mask, setMask] = useState('');
    const [modelLoaded, setModelLoaded] = useState(false);
    const [imageProcessing, setImageProcessing] = useState(false);
    const [selectPoints, setSelectPoints] = useState([]);
    const [selectedImage, setSelectedImage] = useState(null);
    const [originalDimensions, setOriginalDimensions] = useState(null);
    const [displayDimensions, setDisplayDimensions] = useState(null);

    useEffect(() => {
        SAMModule.initializeModels().then(() => setModelLoaded(true));
    }, []);

    useEffect(() => {
        if (originalDimensions) {
            const aspectRatio = originalDimensions.height / originalDimensions.width;
            setDisplayDimensions({ width: DISPLAY_WIDTH, height: DISPLAY_HEIGHT });
        }
    }, [originalDimensions]);

    const pickImage = async () => {
        ImagePicker.launchImageLibrary({
            mediaType: 'photo',
            includeBase64: false,
        }, async (response) => {
            if (response.didCancel || !response.assets?.[0]) return;

            const base64Image = await RNFS.readFile(response.assets[0].uri, 'base64');
            Image.getSize(response.assets[0].uri, (width, height) => {
                setOriginalDimensions({ width, height });
                setSelectedImage(base64Image);
            });
        });
    };

    const handlePress = async (event) => {
        if (!selectedImage || imageProcessing || !displayDimensions) return;
    
        const { locationX, locationY } = event.nativeEvent;
        
        const scaleX = originalDimensions.width / displayDimensions.width;
        const scaleY = originalDimensions.height / displayDimensions.height;
        
        const originalX = locationX * scaleX;
        const originalY = locationY * scaleY;
        
        const normalizedX = originalX / originalDimensions.width;
        const normalizedY = originalY / originalDimensions.height;
        
        setSelectPoints([{ x: locationX, y: locationY }]);
    
        try {
            setImageProcessing(true);
            const maskBase64 = await SAMModule.processImage(
                selectedImage,
                [{
                    x: normalizedX,
                    y: normalizedY,
                    label: 1
                }],
                0
            );
            setMask(maskBase64);
        } catch (error) {
            console.error('Processing failed:', error);
        } finally {
            setImageProcessing(false);
        }
    };

    if (!modelLoaded) {
        return (
            <View style={styles.container}>
                <ActivityIndicator size="large" color="#2B5989" />
                <Text>Loading model...</Text>
            </View>
        );
    }

    if (!selectedImage || !displayDimensions) {
        return (
            <View style={styles.container}>
                <TouchableOpacity onPress={pickImage} style={styles.button}>
                    <Text style={styles.buttonText}>Pick Image</Text>
                </TouchableOpacity>
            </View>
        );
    }


    return (
        <View style={styles.container}>
            <View>
                <TouchableOpacity
                    onPress={handlePress}
                    activeOpacity={1}
                    style={{
                        position: 'relative'
                    }}
                >
                    { mask ? <Image
                        source={{ uri: `data:image/jpeg;base64,${mask}` }}
                        style={{
                            height: 400,
                            width: 400,
                        }}
                        /> : 
                        <Image
                            source={{ uri: `data:image/png;base64,${selectedImage}` }}
                            style={{
                                height: 400,
                                width: 400,
                            }}
                        />
                        }
                    {selectPoints.length >0 && selectPoints.map((point, index) => (
                        <View
                            key={index}
                            style={{
                                position: 'absolute',
                                width: 10,
                                height: 10,
                                borderRadius: 5,
                                backgroundColor: 'red',
                                zIndex: 2,
                                left: point.x,
                                top: point.y,
                            }}
                        />
                    ))}
                </TouchableOpacity>
            </View>

            {imageProcessing && (
                <View style={styles.overlay}>
                    <ActivityIndicator size="large" color="#2B5989" />
                </View>
            )}

            <TouchableOpacity
                style={styles.button}
                onPress={() => {
                    setSelectedImage(null);
                    setMask('');
                    setSelectPoints([]);
                    setDisplayDimensions(null);
                    SAMModule.clearPoints();
                }}
            >
                <Text style={styles.buttonText}>Remove Image</Text>
            </TouchableOpacity>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f5f5f5',
        alignItems: 'center',
        justifyContent: 'center',
    },
    imageWrapper: {
        width: DISPLAY_WIDTH,
        height:DISPLAY_HEIGHT,
        overflow: 'hidden',
        backgroundColor: '#fff',
    },
    image: {
        width: DISPLAY_WIDTH,
        height:DISPLAY_HEIGHT,
        position: 'absolute',
        top: 0,
        left: 0,
    },
    point: {
        position: 'absolute',
        width: 10,
        height: 10,
        borderRadius: 5,
        backgroundColor: 'red',
        zIndex: 2,
    },
    overlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(255,255,255,0.7)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    button: {
        position: 'absolute',
        bottom: 20,
        backgroundColor: '#2B5989',
        paddingHorizontal: 20,
        paddingVertical: 12,
        borderRadius: 8,
    },
    buttonText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
    }
});

export default App;