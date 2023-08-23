const PeerId = require('peer-id')

const createId = async () => {
    const id = await PeerId.create({ bits: 1024, keyType: 'RSA' })
    return id;
}
const createIds = async () => {
    const array = new Array(250)
    for (let index = 0; index < array.length; index++) {
        array[index]=  await createId();
        
    }
    console.log(JSON.stringify(array, null, 2))
} 


createIds()


