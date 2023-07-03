import models from '../../../../models.json'
import { NextResponse } from 'next/server'

export async function GET(re) {
  return NextResponse.json(models)
}
