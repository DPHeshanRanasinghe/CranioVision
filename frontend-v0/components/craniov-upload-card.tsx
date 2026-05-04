'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Upload } from 'lucide-react'

export function UploadCard() {
  const demoCases = [
    'BraTS-GLI-02143-102',
    'BraTS-GLI-02196-105',
    'BraTS-GLI-02105-105',
    'BraTS-GLI-02137-104',
  ]

  return (
    <Card className="border border-gray-200 bg-white">
      <CardHeader>
        <CardTitle>Upload MRI Case</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex items-center justify-center rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 p-12 text-center transition-colors hover:border-blue-400 hover:bg-blue-50">
          <div className="space-y-2">
            <Upload className="mx-auto h-8 w-8 text-gray-400" />
            <p className="text-sm font-medium text-gray-700">Drop a BraTS case folder here, or click to browse</p>
          </div>
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700">Or select a demo case</label>
          <Select>
            <SelectTrigger>
              <SelectValue placeholder="Choose a demo case" />
            </SelectTrigger>
            <SelectContent>
              {demoCases.map((caseId) => (
                <SelectItem key={caseId} value={caseId}>
                  {caseId}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  )
}
