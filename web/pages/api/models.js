import models from '../../../models.json'

export default async function handler(req, res) {
  return res.status(200).json(models)
}
