document.addEventListener('DOMContentLoaded', domReady);

        function domReady() {
            new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[1],
                hideTexts: false,
                textPosition: "bottom"

            });
        }

        // function largeSceneEvent(idx) {
        //     let dics = document.querySelectorAll('.b-dics')[0]
        //     let sections = dics.getElementsByClassName('b-dics__section')
        //     let imagesLength = 3
        //     for (let i = 0; i < imagesLength; i++) {
        //         let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
        //         switch (idx) {
        //             case 0:
        //                 image.src = 'assets/img/nvidia/';
        //                 break;
        //             case 1:
        //                 image.src = 'assets/img/jhu/';
        //                 break;
        //             case 2:
        //                 image.src = 'assets/img/Barn/';
        //                 break;
        //             case 3:
        //                 image.src = 'assets/img/Caterpillar/';
        //                 break;
        //             case 4:
        //                 image.src = 'assets/img/Courthouse/';
        //                 break;
        //             case 5:
        //                 image.src = 'assets/img/Ignatius/';
        //                 break;
        //             case 6:
        //                 image.src = 'assets/img/Meetingroom/';
        //                 break;
        //             case 7:
        //                 image.src = 'assets/img/Truck/';
        //                 break;
        //         }
        //         switch (i) {
        //             case 0:
        //                 image.src = image.src + '/rgb.png';
        //                 break;
        //             case 1:
        //                 image.src = image.src + '/mesh.png';
        //                 break;
        //             case 2:
        //                 image.src = image.src + '/normal.png';
        //                 break;
        //         }
        //     }

        //     scene_list = document.getElementById("large-scale-recon-1").children;
        //     for (let i = 0; i < scene_list.length; i++) {
        //         if (idx == i) {
        //             scene_list[i].children[0].className = "nav-link active"
        //         }
        //         else {
        //             scene_list[i].children[0].className = "nav-link"
        //         }
        //     }
        //     scene_list = document.getElementById("large-scale-recon-2").children;
        //     for (let i = 0; i < scene_list.length; i++) {
        //         if (idx == i+2) {
        //             scene_list[i].children[0].className = "nav-link active"
        //         }
        //         else {
        //             scene_list[i].children[0].className = "nav-link"
        //         }
        //     }
        // }

        // function objectSmallSceneEvent(idx) {
        //     let dics = document.querySelectorAll('.b-dics')[0]
        //     let sections = dics.getElementsByClassName('b-dics__section')
        //     let imagesLength = 4
        //     for (let i = 0; i < imagesLength; i++) {
        //         let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
        //         switch (idx) {
        //             case 0:
        //                 image.src = 'media/small_static_cmp/sky-20/';
        //                 break;
        //             case 1:
        //                 image.src = 'media/small_static_cmp/road-22/';
        //                 break;
        //             case 2:
        //                 image.src = 'media/small_static_cmp/hydrant-14/';
        //                 break;
        //             case 3:
        //                 image.src = 'media/small_static_cmp/lab-16/';
        //                 break;    
        //             case 4:
        //                 image.src = 'media/small_static_cmp/garden-8/';
        //                 break;
        //             case 5:
        //                 image.src = 'media/small_static_cmp/bicycle-1/';
        //                 break;
        //             case 6:
        //                 image.src = 'media/small_static_cmp/room-22/';
        //                 break;
        //             case 7:
        //                 image.src = 'media/small_static_cmp/counter-19/';
        //                 break;
        //         }
        //         switch (i) {
        //             case 0:
        //                 image.src = image.src + 'instant-ngp.png';
        //                 break;
        //             case 1:
        //                 image.src = image.src + 'f2-nerf.png';
        //                 break;
        //             case 2:
        //                 image.src = image.src + 'nelf-pro_pred.png';
        //                 break;
        //             case 3:
        //                 image.src = image.src + 'gt.png';
        //                 break;

        //         }
        //     }

        //     let scene_list = document.getElementById("small-scale-static-cmp").children;
        //     for (let i = 0; i < scene_list.length; i++) {
        //         if (idx == i) {
        //             scene_list[i].children[0].className = "nav-link active"
        //         }
        //         else {
        //             scene_list[i].children[0].className = "nav-link"
        //         }
        //     }
        // }

        function objectSmalleSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 4
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'media/free_static_cmp/sky-14/';
                        break;
                    case 1:
                        image.src = 'media/free_static_cmp/road-22/';
                        break;
                    case 2:
                        image.src = 'media/free_static_cmp/lab-16/';
                        break;
                    case 3:
                        image.src = 'media/free_static_cmp/hydrant-14/';
                        break;    
                    case 4:
                        image.src = 'media/free_static_cmp/pillar-0/';
                        break;
                    case 5:
                        image.src = 'media/free_static_cmp/grass-25/';
                        break;                    
                    case 10:
                        image.src = 'media/360_static_cmp/counter-19/';
                        break;
                    case 11:
                        image.src = 'media/360_static_cmp/room-22/';
                        break;
                    case 12:
                        image.src = 'media/360_static_cmp/bonsai-33/';
                        break;
                    case 13:
                        image.src = 'media/360_static_cmp/garden-8/';
                        break;    
                    case 14:
                        image.src = 'media/360_static_cmp/stump-8/';
                        break;
                    case 15:
                        image.src = 'media/360_static_cmp/bicycle-1/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + 'instant-ngp.png';
                        break;
                    case 1:
                        image.src = image.src + 'f2-nerf.png';
                        break;
                    case 2:
                        image.src = image.src + 'nelf-pro_pred.png';
                        break;
                    case 3:
                        image.src = image.src + 'gt.png';
                        break;

                }
            }

            let scene_list = document.getElementById("free-static-cmp").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }

            scene_list = document.getElementById("360-static-cmp").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i+10) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function objectLargeSceneEvent(idx) {
            console.log(document.querySelectorAll('.b-dics'))
            let dics = document.querySelectorAll('.b-dics')[1]
            console.log(dics)
            let sections = dics.getElementsByClassName('b-dics__section')
            console.log(sections)
            let imagesLength = 2
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'resource/frame24/';
                        break;
                    case 1:
                        image.src = 'media/large_static_cmp/kitti360_long-39/';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '3dgs.png';
                        break;
                    case 1:
                        image.src = image.src + 'ours.png';
                        break;
                }
            }

            let scene_list = document.getElementById("large-scale-static-cmp").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }